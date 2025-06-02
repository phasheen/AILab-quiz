import torch
import torch.nn as nn
import random
import torch.optim as optim
from tqdm import tqdm
import itertools
from .visualization_utils import visualize_generated_images
from pathlib import Path

class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables using older images rather than the latest ones, stabilizing training.
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5: # Use a history image
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else: # Use current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

def define_gan_loss_fn(gan_mode='lsgan', device=None):
    """Returns a GAN loss function (criterion) for a discriminator.
       The function returned will expect (prediction, target_is_real) as input.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if gan_mode == 'vanilla':
        loss_fn = nn.BCEWithLogitsLoss()
    elif gan_mode == 'lsgan':
        loss_fn = nn.MSELoss()
    # elif gan_mode == 'wgangp': # Placeholder for Wasserstein GAN GP
    #     loss_fn = None # WGAN-GP has a different loss structure, often -D(x) for real and D(G(z)) for fake
    else:
        raise NotImplementedError(f'GAN mode {gan_mode} not implemented')

    def criterion(prediction, target_is_real):
        if gan_mode == 'vanilla' or gan_mode == 'lsgan':
            target_val = 1.0 if target_is_real else 0.0
            target_tensor = torch.full_like(prediction, target_val, device=device)
            return loss_fn(prediction, target_tensor)
        # elif gan_mode == 'wgangp':
        #     if target_is_real:
        #         return -prediction.mean()
        #     else:
        #         return prediction.mean()
        else:
             # Should have been caught by NotImplementedError
            return None 
            
    return criterion 

def train_cyclegan_epoch(
    netG_A2B, netG_B2A, netD_A, netD_B,
    dataloader_A, dataloader_B, 
    optimizer_G, optimizer_D,
    criterion_GAN, # Single GAN loss function (e.g., from define_gan_loss_fn)
    criterion_cycle, criterion_identity,
    lambda_cycle, lambda_identity,
    device,
    fake_A_pool, fake_B_pool
):
    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()

    epoch_losses = {
        'G_total': 0, 'G_adv': 0, 'G_cycle': 0, 'G_identity': 0,
        'D_total': 0, 'D_A_real': 0, 'D_A_fake': 0, 'D_B_real': 0, 'D_B_fake': 0
    }
    batch_count = 0

    # Use the shorter dataloader length to avoid issues if datasets are different sizes
    num_batches = min(len(dataloader_A), len(dataloader_B))
    # Create infinite iterators if datasets are of different sizes and full pass is desired (more complex)
    # For simplicity, this example iterates up to the length of the shorter dataset.
    data_iter_A = iter(dataloader_A)
    data_iter_B = iter(dataloader_B)

    progress_bar = tqdm(range(num_batches), desc='CycleGAN Training Epoch')

    for _ in progress_bar:
        # Fetch data. Assume dataloaders yield (image, label) or just (image,)
        # We only need the image for unsupervised CycleGAN.
        try:
            real_A_batch = next(data_iter_A)
            real_A = real_A_batch[0] if isinstance(real_A_batch, (list,tuple)) else real_A_batch
            real_B_batch = next(data_iter_B)
            real_B = real_B_batch[0] if isinstance(real_B_batch, (list,tuple)) else real_B_batch
        except StopIteration:
            # Should not happen if num_batches is min(len(A), len(B))
            break 
            
        real_A, real_B = real_A.to(device), real_B.to(device)

        # --- Train Generators (G_A2B and G_B2A) --- #
        optimizer_G.zero_grad()
        
        # Identity loss (L_identity = lambda_identity * (||G_B2A(A) - A|| + ||G_A2B(B) - B||) )
        loss_G_identity_A, loss_G_identity_B = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        if lambda_identity > 0:
            # G_A2B should be identity if real_B is fed
            identity_B = netG_A2B(real_B)
            loss_G_identity_B = criterion_identity(identity_B, real_B) * lambda_identity
            # G_B2A should be identity if real_A is fed
            identity_A = netG_B2A(real_A)
            loss_G_identity_A = criterion_identity(identity_A, real_A) * lambda_identity

        # GAN loss for G_A2B (tries to fool D_B)
        # L_GAN(G_A2B, D_B, A, B) = E_{a~A}[log D_B(G_A2B(a))] (vanilla) or E_{a~A}[(D_B(G_A2B(a)) - 1)^2] (LSGAN)
        fake_B = netG_A2B(real_A)
        pred_fake_B_for_G = netD_B(fake_B)
        loss_G_A2B_adv = criterion_GAN(pred_fake_B_for_G, True) # Target is real for generator loss

        # GAN loss for G_B2A (tries to fool D_A)
        # L_GAN(G_B2A, D_A, B, A) = E_{b~B}[log D_A(G_B2A(b))] or E_{b~B}[(D_A(G_B2A(b)) - 1)^2]
        fake_A = netG_B2A(real_B)
        pred_fake_A_for_G = netD_A(fake_A)
        loss_G_B2A_adv = criterion_GAN(pred_fake_A_for_G, True) # Target is real for generator loss

        # Cycle consistency loss (L_cyc = lambda_cycle * (||G_B2A(G_A2B(A)) - A|| + ||G_A2B(G_B2A(B)) - B||) )
        recovered_A = netG_B2A(fake_B) # real_A -> fake_B (by G_A2B) -> recovered_A (by G_B2A)
        loss_cycle_A = criterion_cycle(recovered_A, real_A) * lambda_cycle

        recovered_B = netG_A2B(fake_A) # real_B -> fake_A (by G_B2A) -> recovered_B (by G_A2B)
        loss_cycle_B = criterion_cycle(recovered_B, real_B) * lambda_cycle

        # Total generator loss
        loss_G = loss_G_A2B_adv + loss_G_B2A_adv + loss_cycle_A + loss_cycle_B + loss_G_identity_A + loss_G_identity_B
        loss_G.backward()
        optimizer_G.step()

        # --- Train Discriminator D_A --- #
        # L_D_A = 0.5 * (E_{a~A}[ (D_A(a)-1)^2 ] + E_{b~B}[ (D_A(G_B2A(b)))^2 ]) for LSGAN
        optimizer_D.zero_grad()
        
        # Real loss for D_A
        pred_real_A = netD_A(real_A)
        loss_D_A_real = criterion_GAN(pred_real_A, True)

        # Fake loss for D_A
        fake_A_pooled = fake_A_pool.query(fake_A.detach()) # Use .detach() to stop gradients flowing to G_B2A
        pred_fake_A = netD_A(fake_A_pooled)
        loss_D_A_fake = criterion_GAN(pred_fake_A, False)

        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_A.backward() # Accumulate D_A gradients, D_B grads will be separate
        
        # --- Train Discriminator D_B --- #
        # L_D_B = 0.5 * (E_{b~B}[ (D_B(b)-1)^2 ] + E_{a~A}[ (D_B(G_A2B(a)))^2 ]) for LSGAN
        # No optimizer_D.zero_grad() here if D_A and D_B params are in the same optimizer and we want to accumulate
        # However, it's cleaner to step D optimizer once after both D_A and D_B have .backward() called if they are separate parameter groups
        # If they share an optimizer, then this backward() call accumulates with D_A's.

        # Real loss for D_B
        pred_real_B = netD_B(real_B)
        loss_D_B_real = criterion_GAN(pred_real_B, True)

        # Fake loss for D_B
        fake_B_pooled = fake_B_pool.query(fake_B.detach()) # Use .detach() to stop gradients flowing to G_A2B
        pred_fake_B = netD_B(fake_B_pooled)
        loss_D_B_fake = criterion_GAN(pred_fake_B, False)
        
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D_B.backward() # Accumulate D_B gradients

        optimizer_D.step() # Step D optimizer once for both D_A and D_B updates

        # Accumulate losses for logging
        epoch_losses['G_total'] += loss_G.item()
        epoch_losses['G_adv'] += (loss_G_A2B_adv.item() + loss_G_B2A_adv.item())
        epoch_losses['G_cycle'] += (loss_cycle_A.item() + loss_cycle_B.item())
        if lambda_identity > 0:
             epoch_losses['G_identity'] += (loss_G_identity_A.item() + loss_G_identity_B.item())
        epoch_losses['D_total'] += (loss_D_A.item() + loss_D_B.item())
        epoch_losses['D_A_real'] += loss_D_A_real.item()
        epoch_losses['D_A_fake'] += loss_D_A_fake.item()
        epoch_losses['D_B_real'] += loss_D_B_real.item()
        epoch_losses['D_B_fake'] += loss_D_B_fake.item()
        batch_count += 1
        
        progress_bar.set_postfix({
            'G_loss': loss_G.item(), 'D_loss': (loss_D_A.item() + loss_D_B.item())
        })
    
    # Average losses over batches
    for key in epoch_losses: epoch_losses[key] /= batch_count

    # Return losses and a sample of images for visualization
    return epoch_losses, real_A, fake_B, real_B, fake_A 

def train_cyclegan(
    netG_A2B, netG_B2A, netD_A, netD_B,
    dataloader_A, dataloader_B, # Dataloaders for domain A and domain B
    num_epochs, config, # General training parameters from main config
    device,
    output_dir # Base directory to save models and visualizations
):
    # Optimizers
    optimizer_G = optim.Adam(
        itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), 
        lr=config.get('lr_G', 0.0002), 
        betas=tuple(config.get('betas_G', [0.5, 0.999]))
    )
    optimizer_D = optim.Adam(
        itertools.chain(netD_A.parameters(), netD_B.parameters()), 
        lr=config.get('lr_D', 0.0002), 
        betas=tuple(config.get('betas_D', [0.5, 0.999]))
    )

    # Learning rate schedulers (linear decay)
    # n_epochs_decay_start: epoch to start linearly decaying the learning rate to zero
    # n_epochs_total: total number of epochs for training (num_epochs from args)
    # n_epochs_decay: number of epochs to decay over (num_epochs - n_epochs_decay_start)
    n_epochs_decay_start = config.get('n_epochs_decay_start', num_epochs // 2) 
    n_epochs_decay = num_epochs - n_epochs_decay_start

    def lambda_rule(epoch):
        # lr_l = 1.0 - max(0, epoch - n_epochs_decay_start) / float(n_epochs_decay + 1)
        # Corrected LR decay: decay starts *after* n_epochs_decay_start
        if epoch < n_epochs_decay_start:
            return 1.0
        return 1.0 - (epoch - n_epochs_decay_start) / float(n_epochs_decay + 1)

    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    lr_scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    # Image pools
    fake_A_pool = ImagePool(config.get('pool_size', 50))
    fake_B_pool = ImagePool(config.get('pool_size', 50))

    # Loss functions
    # GAN loss (e.g., MSE for LSGAN, BCEWithLogits for vanilla)
    criterion_GAN = define_gan_loss_fn(gan_mode=config.get('gan_mode', 'lsgan'), device=device)
    criterion_cycle = nn.L1Loss() # Cycle consistency loss
    criterion_identity = nn.L1Loss() # Identity loss

    # Loss weights
    lambda_cycle = config.get('lambda_cycle', 10.0)
    lambda_identity = config.get('lambda_identity', 0.5) # Set to 0 if not using identity loss

    # History tracking
    history_keys = ['G_total', 'G_adv', 'G_cycle', 'G_identity', 'D_total', 'D_A_real', 'D_A_fake', 'D_B_real', 'D_B_fake']
    history = {k: [] for k in history_keys}

    # Directories for outputs
    vis_dir = Path(output_dir) / 'cyclegan_visualizations'
    vis_dir.mkdir(exist_ok=True, parents=True)
    model_dir = Path(output_dir) / 'cyclegan_models'
    model_dir.mkdir(exist_ok=True, parents=True)

    print("Starting CycleGAN training...")
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        epoch_losses_map, rA, fB, rB, fA = train_cyclegan_epoch(
            netG_A2B, netG_B2A, netD_A, netD_B,
            dataloader_A, dataloader_B,
            optimizer_G, optimizer_D,
            criterion_GAN, 
            criterion_cycle, criterion_identity,
            lambda_cycle, lambda_identity,
            device,
            fake_A_pool, fake_B_pool
        )

        # Log losses
        for key, loss_val in epoch_losses_map.items():
            history[key].append(loss_val)
        
        # Print epoch summary
        print(f"  G_total: {epoch_losses_map['G_total']:.4f}, D_total: {epoch_losses_map['D_total']:.4f}")
        if lambda_identity > 0:
            print(f"  G_adv: {epoch_losses_map['G_adv']:.4f}, G_cycle: {epoch_losses_map['G_cycle']:.4f}, G_identity: {epoch_losses_map['G_identity']:.4f}")
        else:
            print(f"  G_adv: {epoch_losses_map['G_adv']:.4f}, G_cycle: {epoch_losses_map['G_cycle']:.4f}")
        print(f"  D_A (real/fake): {epoch_losses_map['D_A_real']:.4f}/{epoch_losses_map['D_A_fake']:.4f}, D_B (real/fake): {epoch_losses_map['D_B_real']:.4f}/{epoch_losses_map['D_B_fake']:.4f}")

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        # Visualize some results periodically
        if epoch % config.get('vis_freq_epoch', 1) == 0 and rA is not None and fB is not None:
            vis_batch_size = min(4, rA.size(0)) # Show at most 4 images from each domain for comparison
            if rB is not None and fA is not None:
                 display_batch = torch.cat([rA[:vis_batch_size], fB[:vis_batch_size], rB[:vis_batch_size], fA[:vis_batch_size]], dim=0)
                 num_rows_display = vis_batch_size * 2
            else: # Only A->B for now
                 display_batch = torch.cat([rA[:vis_batch_size], fB[:vis_batch_size]], dim=0)
                 num_rows_display = vis_batch_size
            
            visualize_generated_images(
                display_batch,
                n=display_batch.size(0), nrow=vis_batch_size,
                title=f'Epoch {epoch} (RealA, FakeB, RealB, FakeA)',
                save_path=str(vis_dir / f'epoch_{epoch}_samples.png')
            )

        # Save models periodically
        if epoch % config.get('save_freq_epoch', 5) == 0:
            torch.save(netG_A2B.state_dict(), str(model_dir / f'netG_A2B_epoch_{epoch}.pth'))
            torch.save(netG_B2A.state_dict(), str(model_dir / f'netG_B2A_epoch_{epoch}.pth'))
            torch.save(netD_A.state_dict(), str(model_dir / f'netD_A_epoch_{epoch}.pth'))
            torch.save(netD_B.state_dict(), str(model_dir / f'netD_B_epoch_{epoch}.pth'))
            print(f"Models saved at epoch {epoch} to {model_dir}")

    # Save final models
    torch.save(netG_A2B.state_dict(), str(model_dir / 'netG_A2B_final.pth'))
    torch.save(netG_B2A.state_dict(), str(model_dir / 'netG_B2A_final.pth'))
    torch.save(netD_A.state_dict(), str(model_dir / 'netD_A_final.pth'))
    torch.save(netD_B.state_dict(), str(model_dir / 'netD_B_final.pth'))
    print(f"Final models saved to {model_dir}")
    return history 