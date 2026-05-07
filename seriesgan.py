"""
SeriesGAN — faithful implementation of arXiv:2410.21203
Keras 3 compatible + checkpoint/Drive sync.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from utils import extract_time, random_generator, batch_generator
from metrics.discriminative_metrics import discriminative_score_metrics
from drive_sync import pull_checkpoints, push_checkpoints


def _make_stacked_gru(n_layers, units, return_sequences=True, return_state=False, name_prefix='gru'):
    """Create a list of GRU layers to be applied sequentially (Keras 3 compatible)."""
    layers = []
    for i in range(n_layers):
        rs = return_sequences if i == n_layers - 1 else True
        rst = return_state if i == n_layers - 1 else False
        layers.append(tf.keras.layers.GRU(
            units, return_sequences=rs, return_state=rst,
            activation='tanh', name=f'{name_prefix}_{i}'
        ))
    return layers


def _apply_gru_stack(layers, x):
    """Apply stacked GRU layers, handling return_state on the last layer."""
    state = None
    for layer in layers:
        result = layer(x)
        if isinstance(result, (list, tuple)):
            x, state = result[0], result[1]
        else:
            x = result
    return (x, state) if state is not None else x


def seriesgan(ori_data, parameters, num_samples):
    tf.compat.v1.reset_default_graph()

    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, max_seq_len = extract_time(ori_data)

    # Normalization
    min_val = np.min(np.min(ori_data, axis=0), axis=0)
    ori_data = ori_data - min_val
    max_val = np.max(np.max(ori_data, axis=0), axis=0)
    ori_data = ori_data / (max_val + 1e-7)

    hidden_dim = dim if parameters['hidden_dim'] == 'same' else parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    z_dim = dim
    gamma, beta = 1, 1
    temporal_dimension = 16

    checkpoint_dir   = parameters.get('checkpoint_dir', './seriesgan_checkpoints')
    checkpoint_every = parameters.get('checkpoint_every', 500)
    drive_folder_id  = parameters.get('drive_folder_id', None)
    gdrive_secret    = parameters.get('gdrive_secret', 'GDRIVE_SERVICE_ACCOUNT')
    os.makedirs(checkpoint_dir, exist_ok=True)
    np.savez(os.path.join(checkpoint_dir, 'norm_stats.npz'), min_val=min_val, max_val=max_val)

    # Placeholders
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
    T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")

    # ============== Create all Keras layers (Keras 3 compatible) ==============

    # Temporal embedder (Loss Function AE encoder)
    te_gru = _make_stacked_gru(num_layers, dim, return_sequences=False, return_state=False, name_prefix='te_gru')
    te_dense = tf.keras.layers.Dense(temporal_dimension, name='te_dense')

    # Temporal recovery (Loss Function AE decoder)
    tr_expand = tf.keras.layers.Dense(max_seq_len * dim, name='tr_expand')
    tr_gru = _make_stacked_gru(num_layers, dim, name_prefix='tr_gru')
    tr_dense = tf.keras.layers.Dense(dim, name='tr_dense')

    # Embedder (Latent AE encoder)
    emb_gru = _make_stacked_gru(num_layers, hidden_dim, name_prefix='emb_gru')
    emb_dense = tf.keras.layers.Dense(hidden_dim, name='emb_dense')

    # Recovery (Latent AE decoder)
    rec_gru = _make_stacked_gru(num_layers, hidden_dim, name_prefix='rec_gru')
    rec_dense = tf.keras.layers.Dense(dim, name='rec_dense')

    # Generator
    gen_gru = _make_stacked_gru(num_layers, hidden_dim, name_prefix='gen_gru')
    gen_dense = tf.keras.layers.Dense(hidden_dim, name='gen_dense')

    # Supervisor
    sup_gru = _make_stacked_gru(num_layers - 1, hidden_dim, name_prefix='sup_gru')
    sup_dense = tf.keras.layers.Dense(hidden_dim, name='sup_dense')

    # Latent Discriminator
    dis_gru = _make_stacked_gru(num_layers, hidden_dim, name_prefix='dis_gru')
    dis_dense = tf.keras.layers.Dense(hidden_dim, name='dis_dense')

    # Feature Discriminator (AE discriminator)
    aed_gru = _make_stacked_gru(num_layers, hidden_dim, name_prefix='aed_gru')
    aed_flat = tf.keras.layers.Flatten(name='aed_flat')
    aed_dense = tf.keras.layers.Dense(hidden_dim, name='aed_dense')

    # ============== Network functions ==============

    def f_temporal_embedder(inp):
        h = _apply_gru_stack(te_gru, inp)
        return te_dense(h)

    def f_temporal_recovery(h_t):
        expanded = tr_expand(h_t)
        expanded = tf.reshape(expanded, [-1, max_seq_len, dim])
        out = _apply_gru_stack(tr_gru, expanded)
        return tr_dense(out)

    def f_embedder(inp):
        out = _apply_gru_stack(emb_gru, inp)
        return emb_dense(out)

    def f_recovery(h):
        out = _apply_gru_stack(rec_gru, h)
        return rec_dense(out)

    def f_generator(z):
        out = _apply_gru_stack(gen_gru, z)
        return gen_dense(out)

    def f_supervisor(h):
        out = _apply_gru_stack(sup_gru, h)
        return sup_dense(out)

    def f_discriminator(h):
        out = _apply_gru_stack(dis_gru, h)
        return dis_dense(out)

    def f_ae_discriminator(inp):
        out = _apply_gru_stack(aed_gru, inp)
        flat = aed_flat(out)
        return aed_dense(flat)

    # ============== Build Computation Graph ==============

    H = f_embedder(X)
    X_tilde = f_recovery(H)
    Y_ae_fake = f_ae_discriminator(X_tilde)
    Y_ae_real = f_ae_discriminator(X)

    E_hat = f_generator(Z)
    H_hat = f_supervisor(E_hat)
    H_hat_supervise = f_supervisor(H)

    X_hat = f_recovery(H_hat)
    Y_ae_fake_e = f_ae_discriminator(X_hat)
    X_tilde_fake_second = f_recovery(E_hat)
    Y_ae_fake_e_second = f_ae_discriminator(X_tilde_fake_second)

    Y_fake = f_discriminator(H_hat)
    Y_real = f_discriminator(H)
    Y_fake_e = f_discriminator(E_hat)

    H_t = f_temporal_embedder(X)
    X_t = f_temporal_recovery(H_t)
    H_t_hat = f_temporal_embedder(X_hat)

    # ============== Collect trainable variables by layer prefix ==============

    all_vars = tf.compat.v1.trainable_variables()
    e_t_vars = [v for v in all_vars if 'te_' in v.name]
    r_t_vars = [v for v in all_vars if 'tr_' in v.name]
    e_vars   = [v for v in all_vars if 'emb_' in v.name]
    r_vars   = [v for v in all_vars if 'rec_' in v.name]
    d_ae_vars = [v for v in all_vars if 'aed_' in v.name]
    g_vars   = [v for v in all_vars if 'gen_' in v.name]
    s_vars   = [v for v in all_vars if 'sup_' in v.name]
    d_vars   = [v for v in all_vars if 'dis_' in v.name and 'aed_' not in v.name]

    # ============== Loss Functions (LSGAN) ==============

    D_loss_real   = tf.reduce_mean(tf.math.squared_difference(Y_real, tf.ones_like(Y_real)))
    D_loss_fake   = tf.reduce_mean(tf.square(Y_fake))
    D_loss_fake_e = tf.reduce_mean(tf.square(Y_fake_e))
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    D_ae_loss_real          = tf.reduce_mean(tf.math.squared_difference(Y_ae_real, tf.ones_like(Y_ae_real)))
    D_ae_loss_fake          = tf.reduce_mean(tf.square(Y_ae_fake))
    D_ae_loss_fake_e        = tf.reduce_mean(tf.square(Y_ae_fake_e))
    D_ae_loss_fake_e_second = tf.reduce_mean(tf.square(Y_ae_fake_e_second))
    D_ae_loss = D_ae_loss_real + D_ae_loss_fake
    D_ae_loss_real_second = tf.reduce_mean(tf.math.squared_difference(Y_ae_fake, tf.ones_like(Y_ae_fake)))
    D_ae_loss_second = D_ae_loss_real + D_ae_loss_real_second + beta * (D_ae_loss_fake_e + gamma * D_ae_loss_fake_e_second)

    G_loss_U      = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_fake), Y_fake))
    G_loss_U_e    = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_fake_e), Y_fake_e))
    G_loss_U_ae   = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_ae_fake_e), Y_ae_fake_e))
    G_loss_U_ae_e = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_ae_fake_e_second), Y_ae_fake_e_second))
    G_loss_U_totall = G_loss_U + G_loss_U_e + G_loss_U_ae + G_loss_U_ae_e

    # 2-step-ahead supervised loss (paper's key contribution)
    G_loss_S = tf.reduce_mean(tf.math.squared_difference(H[:, 2:, :], H_hat_supervise[:, :-2, :]))

    # L_TS: Time Series Characteristics loss
    mean_H_t     = tf.reduce_mean(H_t, axis=0)
    mean_H_t_hat = tf.reduce_mean(H_t_hat, axis=0)
    mse_mean     = tf.reduce_mean(tf.square(mean_H_t - mean_H_t_hat))
    std_H_t      = tf.math.reduce_std(H_t, axis=0)
    std_H_t_hat  = tf.math.reduce_std(H_t_hat, axis=0)
    mse_std      = tf.reduce_mean(tf.square(std_H_t - std_H_t_hat))
    G_loss_ts    = mse_mean + mse_std

    # L_V: Moment matching
    G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(tf.abs(tf.nn.moments(X_hat, [0])[0] - tf.nn.moments(X, [0])[0]))
    G_loss_V  = G_loss_V1 + G_loss_V2

    # Combined Generator loss (Eq. 3)
    G_loss = (G_loss_U + gamma * G_loss_U_e
              + beta * (G_loss_U_ae + gamma * G_loss_U_ae_e)
              + 20 * tf.sqrt(G_loss_S) + 10 * G_loss_V + 20 * G_loss_ts)

    # Embedder loss
    lambda_c = 0.001
    E_loss_T00 = tf.reduce_mean(tf.math.squared_difference(X, X_tilde))
    E_loss_U_emb   = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_ae_fake), Y_ae_fake))
    E_loss_U_e_emb = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_ae_fake_e), Y_ae_fake_e))
    E_loss_U_e_second_emb = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_ae_fake_e_second), Y_ae_fake_e_second))
    E_loss_T0 = E_loss_T00 + lambda_c * E_loss_U_emb
    E_loss_T0_second = E_loss_T00 + 0.1 * (lambda_c * E_loss_U_emb + lambda_c * beta * 0.1 * (E_loss_U_e_emb + gamma * E_loss_U_e_second_emb))

    # ================================================================
    # Contrastive Representation Learning Loss (NT-Xent)
    # ================================================================
    # Create two augmented views of the input data
    X_aug1 = X + tf.random.normal(tf.shape(X), mean=0.0, stddev=0.05)
    X_aug2 = X * tf.random.uniform(tf.shape(X), minval=0.9, maxval=1.1)

    # Encode both views into the latent space
    H1 = f_embedder(X_aug1)
    H2 = f_embedder(X_aug2)

    # Global average pooling over time
    z1 = tf.reduce_mean(H1, axis=1)
    z2 = tf.reduce_mean(H2, axis=1)

    # L2 Normalize
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    # Compute similarity matrix scaled by temperature
    tau = 0.1
    sim_matrix = tf.matmul(z1, z2, transpose_b=True) / tau

    # The positive pairs are on the diagonal
    labels = tf.range(tf.shape(z1)[0])

    # Cross entropy loss
    loss_a = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, sim_matrix)
    loss_b = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, tf.transpose(sim_matrix))
    contrastive_loss = (loss_a + loss_b) / 2.0

    # Add contrastive loss to Embedder
    lambda_contrastive = 0.1
    E_loss0 = tf.sqrt(E_loss_T0) + lambda_contrastive * contrastive_loss
    E_loss  = tf.sqrt(E_loss_T0_second) + 0.01 * G_loss_S + lambda_contrastive * contrastive_loss

    E_loss_temporal = tf.reduce_mean(tf.math.squared_difference(X, X_t))

    # ============== Optimizers ==============

    E_solver_temporal  = tf.compat.v1.train.AdamOptimizer().minimize(E_loss_temporal, var_list=e_t_vars + r_t_vars)
    E0_solver          = tf.compat.v1.train.AdamOptimizer().minimize(E_loss0, var_list=e_vars + r_vars)
    E_solver           = tf.compat.v1.train.AdamOptimizer().minimize(E_loss, var_list=e_vars + r_vars)
    D_ae_solver        = tf.compat.v1.train.AdamOptimizer().minimize(D_ae_loss, var_list=d_ae_vars)
    D_ae_solver_second = tf.compat.v1.train.AdamOptimizer().minimize(D_ae_loss_second, var_list=d_ae_vars)
    D_solver           = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
    G_solver           = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=g_vars + s_vars)
    GS_solver          = tf.compat.v1.train.AdamOptimizer().minimize(G_loss_S, var_list=g_vars + s_vars)

    # ============== Checkpoint ==============

    saver = tf.compat.v1.train.Saver(max_to_keep=20)
    ckpt_prefix = os.path.join(checkpoint_dir, 'seriesgan_ckpt')
    pull_checkpoints(checkpoint_dir, drive_folder_id, gdrive_secret)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        saver.restore(sess, latest_ckpt)
        print(f'[Checkpoint] Restored from {latest_ckpt}')

    final_generated = []
    global_summing = 5

    # ============== Phase 1: Loss Function AE (0.5x iterations) ==============
    p1 = int(iterations * 0.5)
    print(f'Phase 1: Loss Function AE Training ({p1} iters)')
    for itt in range(p1):
        for _ in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            _, step_loss = sess.run([E_solver_temporal, E_loss_temporal], feed_dict={X: X_mb, T: T_mb})
        if itt % 500 == 0 or itt == p1 - 1:
            print(f'  step: {itt*2}/{iterations}, AE_loss: {np.round(step_loss, 4)}')
    print('Phase 1 Complete')

    # ============== Phase 2: Embedding Network (0.5x iterations) ==============
    p2 = int(iterations * 0.5)
    print(f'Phase 2: Embedding Network Training ({p2} iters)')
    step_d_ae = 0.0
    for itt in range(p2):
        for _ in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            _, step_loss = sess.run([E0_solver, E_loss0], feed_dict={X: X_mb, T: T_mb})
        chk = sess.run(D_ae_loss, feed_dict={X: X_mb, T: T_mb})
        if chk > 0.15:
            _, step_d_ae = sess.run([D_ae_solver, D_ae_loss], feed_dict={X: X_mb, T: T_mb})
        if itt % 500 == 0 or itt == p2 - 1:
            print(f'  step: {itt*2}/{iterations}, AE_loss: {np.round(step_loss, 4)}, AE_D: {np.round(step_d_ae, 4)}')
    print('Phase 2 Complete')

    # ============== Phase 3: Supervised Only (1x iterations) ==============
    print(f'Phase 3: Supervised Loss Only ({iterations} iters)')
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        _, step_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
        if itt % 1000 == 0 or itt == iterations - 1:
            print(f'  step: {itt}/{iterations}, S_loss: {np.round(step_s, 4)}')
    print('Phase 3 Complete')

    save_path = saver.save(sess, ckpt_prefix, global_step=0)
    print(f'[Checkpoint] Pre-training saved -> {save_path}')
    push_checkpoints(checkpoint_dir, drive_folder_id, gdrive_secret)

    # ============== Phase 4: Joint Training (1x iterations) ==============
    print(f'Phase 4: Joint Training ({iterations} iters)')
    step_d, step_d_ae = 0.0, 0.0
    for itt in range(iterations):
        for _ in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            _, sg_u, sg_s, sg, sg_ts = sess.run(
                [G_solver, G_loss_U_totall, G_loss_S, G_loss, G_loss_ts],
                feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
            _, se = sess.run([E_solver, E_loss], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})

        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        chk_d = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        if chk_d > 0.15:
            _, step_d = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        chk_ae = sess.run(D_ae_loss_second, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        if chk_ae > 0.15:
            _, step_d_ae = sess.run([D_ae_solver_second, D_ae_loss_second], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})

        if itt % 1000 == 0 or itt == iterations - 1:
            print(f'  step: {itt}/{iterations}, D: {np.round(step_d,4)}, G: {np.round(sg,4)}'
                  f', G_u: {np.round(sg_u,4)}, G_s: {np.round(sg_s,4)}'
                  f', G_ts: {np.round(sg_ts,4)}, AE: {np.round(se,4)}, AE_D: {np.round(step_d_ae,4)}')

        if (itt + 1) % checkpoint_every == 0:
            sp = saver.save(sess, ckpt_prefix, global_step=itt + 1)
            print(f'[Checkpoint] Saved step {itt+1} -> {sp}')
            push_checkpoints(checkpoint_dir, drive_folder_id, gdrive_secret)

        # Early stopping (after halfway, every 500 steps)
        if (itt >= int(iterations * 0.5)) and (itt % 500 == 0 or itt == iterations - 1):
            Z_mb_g = random_generator(no, z_dim, ori_time, max_seq_len)
            gen_curr = sess.run(X_hat, feed_dict={Z: Z_mb_g, X: ori_data, T: ori_time})
            gen_data = [gen_curr[i, :ori_time[i], :] for i in range(no)]
            gen_data = gen_data * max_val + min_val
            disc_scores = [discriminative_score_metrics(ori_data, gen_data) for _ in range(6)]
            score = np.round(np.min(disc_scores), 4)
            if score <= global_summing:
                global_summing = score
                final_generated = gen_data
                sp = saver.save(sess, ckpt_prefix + '_best', global_step=itt)
                print(f'  [EarlyStop] Best score={score} step {itt} -> {sp}')
                push_checkpoints(checkpoint_dir, drive_folder_id, gdrive_secret)

    sp = saver.save(sess, ckpt_prefix, global_step=iterations)
    print(f'[Checkpoint] Final -> {sp}')
    push_checkpoints(checkpoint_dir, drive_folder_id, gdrive_secret)
    print('Finish Training')

    # ============== Generate ==============
    if num_samples == "same":
        return final_generated
    count = max(1, int(num_samples / no))
    all_gen = []
    for _ in range(count):
        Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
        gen_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})
        gen_data = [gen_curr[i, :ori_time[i], :] for i in range(no)]
        gen_data = gen_data * max_val + min_val
        all_gen.append(gen_data)
    return np.concatenate(all_gen)
