"""
Physical constraints that directly modify optimizable tensors with specified intervals of iterations

"""

import torch
from torch.fft import fft2, fftfreq, fftn, ifft2, ifftn
from torch.nn.functional import interpolate
from torchvision.transforms.functional import gaussian_blur

from ptyrad.utils import fftshift2, gaussian_blur_1d, ifftshift2, make_sigmoid_mask, vprint


@torch.compiler.disable # TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function <Wrapped method <original sub>>(*(FakeTensor(..., size=(5,), dtype=torch.float64), 2.0), **{}): got AttributeError("'ndarray' object has no attribute 'sub'")
class CombinedConstraint(torch.nn.Module):
    """Applies iteration-wise in-place constraints on optimizable tensors.

    This class is designed to apply various constraints to a model's parameters during the optimization process.
    The constraints are applied at specific iteration frequencies, as determined by the `constraint_params` dictionary.
    These constraints include orthogonality, probe amplitude constraints in Fourier space, intensity constraints, Gaussian
    blurring, Fourier filtering, and more.

    Args:
        constraint_params (dict): A dictionary containing the configuration for each constraint. Each constraint should have a 
            frequency and other parameters necessary for its application.
        device (str, optional): The device on which the tensors are located (e.g., 'cuda' or 'cpu'). Defaults to 'cuda'.
        verbose (bool, optional): If True, prints messages during the application of constraints. Defaults to True.
    """
    def __init__(self, constraint_params, device='cuda', verbose=True):
        super(CombinedConstraint, self).__init__()
        self.device = device
        self.constraint_params = constraint_params
        self.verbose = verbose

    def apply_ortho_pmode(self, model, niter):
        ''' Apply orthogonality constraint to probe modes '''
        ortho_pmode_freq = self.constraint_params['ortho_pmode']['freq']
        if ortho_pmode_freq is not None and niter % ortho_pmode_freq == 0:
            model.opt_probe.data = torch.view_as_real(orthogonalize_modes_vec(model.get_complex_probe_view(), sort=True).contiguous()) # Note that model stores the complex probe as a (pmode, Ny, Nx, 2) float tensor (real view) so we need to do some real-complex view conversion.
            probe_int = model.get_complex_probe_view().abs().pow(2)
            probe_pow = (probe_int.sum((1,2))/probe_int.sum()).detach().cpu().numpy().round(3)
            vprint(f"Apply ortho pmode constraint at iter {niter}, relative pmode power = {probe_pow}, probe int sum = {probe_int.sum():.4f}", verbose=self.verbose)

    def apply_probe_mask_k(self, model, niter):
        ''' Apply probe amplitude constraint in Fourier space '''
        # Note that this will change the total probe intensity, please use this with `fix_probe_int`
        # Although the mask wouldn't change during the iteration, making a mask takes only ~0.5us on CPU so really no need to pre-calculate it
        # The sandwitch fftshift(fft(ifftshift(probe))) is needed to properly handle the complex probe without serrated phase
        # fft2 is for real->fourier, while fftshift2 is for corner->center
        
        probe_mask_k_freq = self.constraint_params['probe_mask_k']['freq']
        relative_radius   = self.constraint_params['probe_mask_k']['radius']
        relative_width    = self.constraint_params['probe_mask_k']['width']
        power_thresh      = self.constraint_params['probe_mask_k']['power_thresh']
        if probe_mask_k_freq is not None and niter % probe_mask_k_freq == 0:
            probe = model.get_complex_probe_view()
            Npix = probe.size(-1)
            powers = probe.abs().pow(2).sum((-2,-1)) / probe.abs().pow(2).sum()
            powers_cumsum = powers.cumsum(0)
            pmode_index = (powers_cumsum > power_thresh).nonzero()[0].item() # This gives the pmode index that the cumulative power along mode dimension is greater than the power_thresh and should have mask extend to this index
            mask = torch.ones_like(probe, dtype=torch.float32, device=model.device)
            mask_value = make_sigmoid_mask(Npix, relative_radius, relative_width).to(model.device)
            mask[:pmode_index+1] = mask_value
            probe_k = fftshift2 (fft2(ifftshift2(probe), norm='ortho')) # probe_k at center for later masking
            probe_r = fftshift2(ifft2(ifftshift2(mask * probe_k),  norm='ortho')) # probe_r at center. Note that the norm='ortho' is explicitly specified but not needed for a round-trip
            probe_int = model.get_complex_probe_view().abs().pow(2)
            # Re-sort the probe modes, note that the masked strong modes might be swapping order with unmasked weak modes
            model.opt_probe.data = torch.view_as_real(sort_by_mode_int(probe_r))
            vprint(f"Apply Fourier-space probe amplitude constraint at iter {niter}, pmode_index = {pmode_index} when power_thresh = {power_thresh}, probe int sum = {probe_int.sum():.4f}", verbose=self.verbose)

    def apply_probe_mask_r(self, model, niter):
        ''' Apply probe amplitude constraint in real space '''
        # Note that this will change the total probe intensity, please use this with `fix_probe_int`
        # Although the mask wouldn't change during the iteration, making a mask takes only ~0.5us on CPU so really no need to pre-calculate it
        # The sandwitch fftshift(fft(ifftshift(probe))) is needed to properly handle the complex probe without serrated phase
        # fft2 is for real->fourier, while fftshift2 is for corner->center

        probe_mask_r_freq = self.constraint_params['probe_mask_r']['freq']
        relative_radius   = self.constraint_params['probe_mask_r']['radius']
        relative_width    = self.constraint_params['probe_mask_r']['width']
        power_thresh      = self.constraint_params['probe_mask_r']['power_thresh']
        if probe_mask_r_freq is not None and niter % probe_mask_r_freq == 0:
            probe = model.get_complex_probe_view()
            Npix = probe.size(-1)
            powers = probe.abs().pow(2).sum((-2,-1)) / probe.abs().pow(2).sum()
            powers_cumsum = powers.cumsum(0)
            pmode_index = (powers_cumsum > power_thresh).nonzero()[0].item() # This gives the pmode index that the cumulative power along mode dimension is greater than the power_thresh and should have mask extend to this index
            mask = torch.ones_like(probe, dtype=torch.float32, device=model.device)
            mask_value = make_sigmoid_mask(Npix, relative_radius, relative_width).to(model.device)
            mask[:pmode_index+1] = mask_value
            probe_r = mask * probe # probe_r at center. Note that the norm='ortho' is explicitly specified but not needed for a round-trip
            probe_int = model.get_complex_probe_view().abs().pow(2)
            # Re-sort the probe modes, note that the masked strong modes might be swapping order with unmasked weak modes
            model.opt_probe.data = torch.view_as_real(sort_by_mode_int(probe_r))
            vprint(f"Apply real-space probe amplitude constraint at iter {niter}, pmode_index = {pmode_index} when power_thresh = {power_thresh}, probe int sum = {probe_int.sum():.4f}", verbose=self.verbose)
    
    def apply_fix_probe_int(self, model, niter):
        ''' Apply probe intensity constraint '''
        # Note that the probe intensity fluctuation (std/mean) is typically only 0.5%, there's very little point to do a position-dependent probe intensity constraint
        # Therefore, a mean probe intensity is used here as the target intensity
        fix_probe_int_freq = self.constraint_params['fix_probe_int']['freq']
        if fix_probe_int_freq is not None and niter % fix_probe_int_freq == 0:
            probe = model.get_complex_probe_view()
            current_amp = probe.abs().pow(2).sum().pow(0.5)
            target_amp  = model.probe_int_sum**0.5   
            model.opt_probe.data = torch.view_as_real(probe * target_amp/current_amp)
            probe_int = model.get_complex_probe_view().abs().pow(2)
            vprint(f"Apply fix probe int constraint at iter {niter}, probe int sum = {probe_int.sum():.4f}", verbose=self.verbose)
            
    def apply_obj_rblur(self, model, niter):
        ''' Apply Gaussian blur to object, this only applies to the last 2 dimension (...,H,W) '''
        # Note that it's not clear whether applying blurring after every iteration would ever reach a steady state
        # However, this is at least similar to PtychoShelves' eng. reg_mu
        obj_rblur_freq = self.constraint_params['obj_rblur']['freq']
        obj_type       = self.constraint_params['obj_rblur']['obj_type']
        obj_rblur_ks   = self.constraint_params['obj_rblur']['kernel_size']
        obj_rblur_std  = self.constraint_params['obj_rblur']['std']
        
        if obj_rblur_freq is not None and niter % obj_rblur_freq == 0 and obj_rblur_std !=0:
            if obj_type in ['amplitude', 'both']:
                model.opt_obja.data = gaussian_blur(model.opt_obja, kernel_size=obj_rblur_ks, sigma=obj_rblur_std)
                vprint(f"Apply lateral (y,x) Gaussian blur with std = {obj_rblur_std} px on obja at iter {niter}", verbose=self.verbose)
            if obj_type in ['phase', 'both']:
                model.opt_objp.data = gaussian_blur(model.opt_objp, kernel_size=obj_rblur_ks, sigma=obj_rblur_std)
                vprint(f"Apply lateral (y,x) Gaussian blur with std = {obj_rblur_std} px on objp at iter {niter}", verbose=self.verbose)
    
    def apply_obj_zblur(self, model, niter):
        ''' Apply Gaussian blur to object, this only applies to the last dimension (...,L) '''
        obj_zblur_freq = self.constraint_params['obj_zblur']['freq']
        obj_type       = self.constraint_params['obj_zblur']['obj_type']
        obj_zblur_ks   = self.constraint_params['obj_zblur']['kernel_size']
        obj_zblur_std  = self.constraint_params['obj_zblur']['std']
        if obj_zblur_freq is not None and niter % obj_zblur_freq == 0 and obj_zblur_std !=0:
            if obj_type in ['amplitude', 'both']:
                tensor = model.opt_obja.permute(0,2,3,1)
                model.opt_obja.data = gaussian_blur_1d(tensor, kernel_size=obj_zblur_ks, sigma=obj_zblur_std).permute(0,3,1,2).contiguous() # contiguous() returns a contiguous memory layout so that DDP wouldn't complain about the stride mismatch of grad and params
                vprint(f"Apply z-direction Gaussian blur with std = {obj_zblur_std} px on obja at iter {niter}", verbose=self.verbose)
            if obj_type in ['phase', 'both']:
                tensor = model.opt_objp.permute(0,2,3,1)
                model.opt_objp.data = gaussian_blur_1d(tensor, kernel_size=obj_zblur_ks, sigma=obj_zblur_std).permute(0,3,1,2).contiguous() 
                vprint(f"Apply z-direction Gaussian blur with std = {obj_zblur_std} px on objp at iter {niter}", verbose=self.verbose)
    
    def apply_kr_filter(self, model, niter):
        ''' Apply kr Fourier filter constraint on object '''
        # Note that the `kr_filter` is applied on stacked 2D FFT of object, so it's applying on (omode,z,ky,kx)
        # The kr filter is similar to a top-hat, so it's more like a cut-off, instead of the weak lateral Gaussian blurring (alpha) included in the `kz_filter` 
        kr_filter_freq   = self.constraint_params['kr_filter']['freq']
        obj_type         = self.constraint_params['kr_filter']['obj_type']
        relative_radius  = self.constraint_params['kr_filter']['radius']
        relative_width   = self.constraint_params['kr_filter']['width']
        relative_radius *=  (niter) ** self.constraint_params['kr_filter']['pow']
        if kr_filter_freq is not None and niter % kr_filter_freq == 0:
            if obj_type in ['amplitude', 'both']:
                model.opt_obja.data = kr_filter(model.opt_obja, relative_radius, relative_width)
                vprint(f"Apply kr_filter constraint with kr_radius = {relative_radius} on obja at iter {niter}", verbose=self.verbose)
            if obj_type in ['phase', 'both']:
                model.opt_objp.data = kr_filter(model.opt_objp, relative_radius, relative_width)
                vprint(f"Apply kr_filter constraint with kr_radius = {relative_radius} on objp at iter {niter}", verbose=self.verbose)
                
    def apply_kz_filter(self, model, niter):
        ''' Apply kz Fourier filter constraint on object '''
        # Note that the `kz_filter`` behaves differently for 'amplitude' and 'phase', see `kz_filter` implementaion for details
        kz_filter_freq         = self.constraint_params['kz_filter']['freq']
        obj_type               = self.constraint_params['kz_filter']['obj_type']
        beta_regularize_layers = self.constraint_params['kz_filter']['beta']
        alpha_gaussian         = self.constraint_params['kz_filter']['alpha']
        if kz_filter_freq is not None and niter % kz_filter_freq == 0:
            if obj_type in ['amplitude', 'both']:
                model.opt_obja.data = kz_filter(model.opt_obja, beta_regularize_layers, alpha_gaussian, obj_type='amplitude')
                vprint(f"Apply kz_filter constraint with beta = {beta_regularize_layers} on obja at iter {niter}", verbose=self.verbose)
            if obj_type in ['phase', 'both']:
                model.opt_objp.data = kz_filter(model.opt_objp, beta_regularize_layers, alpha_gaussian, obj_type='phase')
                vprint(f"Apply kz_filter constraint with beta = {beta_regularize_layers} on objp at iter {niter}", verbose=self.verbose)
    
    def apply_complex_ratio(self, model, niter):
        ''' Apply complex constraint on object '''
        # Original paper seems to apply this constraint at each position. I'll try an iteration-wise constraint first
        complex_ratio_freq = self.constraint_params['complex_ratio']['freq']
        obj_type           = self.constraint_params['complex_ratio']['obj_type']
        alpha1             = self.constraint_params['complex_ratio']['alpha1']
        alpha2             = self.constraint_params['complex_ratio']['alpha2']
        if complex_ratio_freq is not None and niter % complex_ratio_freq == 0:
            objac, objpc, Cbar = complex_ratio_constraint(model, alpha1, alpha2)
            if obj_type in ['amplitude', 'both']:
                model.opt_obja.data = objac
                amin, amax = model.opt_obja.min().item(), model.opt_obja.max().item()
                vprint(f"Apply complex ratio constraint with alpha1: {alpha1}, alpha2: {alpha2}, and Cbar: {Cbar.item():.3f} on obja at iter {niter}. obja range becomes ({amin:.3f}, {amax:.3f})", verbose=self.verbose)
            if obj_type in ['phase', 'both']:
                model.opt_objp.data = objpc
                pmin, pmax = model.opt_objp.min().item(), model.opt_objp.max().item()
                vprint(f"Apply complex ratio constraint with alpha1: {alpha1}, alpha2: {alpha2}, and Cbar: {Cbar.item():.3f} on objp at iter {niter}. objp range becomes ({pmin:.3f}, {pmax:.3f})", verbose=self.verbose)  
    
    def apply_mirrored_amp(self, model, niter):
        '''Apply mirrored amplitude constraint on obja at voxel level'''
        # The idea is to replace the amplitude with Amp' = exp(-scale*phase^2), because the absorptive potential should scale with V^2
        mirrored_amp_freq = self.constraint_params['mirrored_amp']['freq']
        relax            = self.constraint_params['mirrored_amp']['relax']
        scale           = self.constraint_params['mirrored_amp']['scale']
        power           = self.constraint_params['mirrored_amp']['power']
        if mirrored_amp_freq is not None and niter % mirrored_amp_freq == 0:
            v_power = model.opt_objp.clamp(min=0).pow(power)
            # amp_new = torch.exp(-scale*v_power)
            amp_new = 1-scale*v_power
            model.opt_obja.data = relax * model.opt_obja + (1-relax) * amp_new
            amin, amax = model.opt_obja.min().item(), model.opt_obja.max().item()
            relax_str = f'relaxed ({relax}*obj + ({1-relax}*obj_new))' if relax != 0 else 'hard'
            vprint(f"Apply {relax_str} mirrored amplitude constraint with scale = {scale} and power = {power} on obja at iter {niter}. obja range becomes ({amin:.3f}, {amax:.3f})", verbose=self.verbose)
    
    def apply_obja_thresh(self, model, niter):
        ''' Apply thresholding on obja at voxel level '''
        # Although there's a lot of code repitition with `apply_postiv`, phase positivity itself is important enough as an individual operation
        obja_thresh_freq = self.constraint_params['obja_thresh']['freq']
        relax            = self.constraint_params['obja_thresh']['relax']
        thresh           = self.constraint_params['obja_thresh']['thresh']
        if obja_thresh_freq is not None and niter % obja_thresh_freq == 0: 
            model.opt_obja.data = relax * model.opt_obja + (1-relax) * model.opt_obja.clamp(min=thresh[0], max=thresh[1])
            relax_str = f'relaxed ({relax}*obj + ({1-relax}*obj_clamp))' if relax != 0 else 'hard'
            vprint(f"Apply {relax_str} threshold constraint with thresh = {thresh} on obja at iter {niter}", verbose=self.verbose)

    def apply_objp_postiv(self, model, niter):
        ''' Apply positivity constraint on objp at voxel level '''
        # Note that this `relax` is defined oppositly to PtychoShelves's `positivity_constraint_object` in `ptycho_solver`. 
        # Here, relax=1 means fully relaxed and essentially no constraint.
        objp_postiv_freq = self.constraint_params['objp_postiv']['freq']
        relax            = self.constraint_params['objp_postiv']['relax']
        mode             = self.constraint_params['objp_postiv'].get('mode', 'clip_neg')
        if objp_postiv_freq is not None and niter % objp_postiv_freq == 0:
            original_min = model.opt_objp.min()
            if mode == 'subtract_min':
                modified_objp = model.opt_objp - original_min
            else: # 'clip_neg'
                modified_objp = model.opt_objp.clamp(min=0)
            model.opt_objp.data = relax * model.opt_objp + (1-relax) * modified_objp
            omin, omax = model.opt_objp.min().item(), model.opt_objp.max().item()
            relax_str = f'relaxed ({relax}*obj + ({1-relax}*obj_postiv))' if relax != 0 else 'hard'
            vprint(f"Apply {relax_str} positivity constraint on objp with '{mode}' mode at iter {niter}. Original min = {original_min.item():.3f}. objp range becomes ({omin:.3f}, {omax:.3f})", verbose=self.verbose)           

    def apply_tilt_smooth(self, model, niter):
        ''' Apply Gaussian blur to object tilts '''
        # Note that the smoothing is applied along the last 2 axes, which are scan dimensions, so the unit of std is "scan positions"
        # Besides, the relative position of the obj_tilts are neglected for simplicity
        tilt_smooth_freq = self.constraint_params['tilt_smooth']['freq']
        tilt_smooth_std  = self.constraint_params['tilt_smooth']['std']
        N_scan_slow = model.N_scan_slow
        N_scan_fast = model.N_scan_fast
        
        if tilt_smooth_freq is not None and niter % tilt_smooth_freq == 0 and tilt_smooth_std !=0:
            if model.opt_obj_tilts.shape[0] == 1: # obj_tilts.shape = (1,2) for tilt_type: 'all', and (N,2) for 'each'
                vprint("`tilt_smooth` constraint requires `tilt_type':'each'`, skip this constraint", verbose=self.verbose)
                return 
            obj_tilts = (model.opt_obj_tilts.reshape(N_scan_slow, N_scan_fast, 2)).permute(2,0,1)
            model.opt_obj_tilts.data = gaussian_blur(obj_tilts, kernel_size=5, sigma=tilt_smooth_std).permute(1,2,0).reshape(-1,2).contiguous() # contiguous() returns a contiguous memory layout so that DDP wouldn't complain about the stride mismatch of grad and params
            vprint(f"Apply Gaussian blur with std = {tilt_smooth_std} scan positions on obj_tilts at iter {niter}", verbose=self.verbose)
    
    def apply_tv_denoise_chambolle(self, model, niter):
        ''' Apply total-variation denoising on object '''
        # Note that the `denoise_tv_chambolle_pytorch` is applied on stacked 2D object, so it's applying on (omode,z,ky,kx)
        tv_denoise_freq = self.constraint_params['tv_denoise_chambolle']['freq']
        weight          = self.constraint_params['tv_denoise_chambolle']['weight']
        if tv_denoise_freq is not None and niter % tv_denoise_freq == 0:
            model.opt_obja.data = denoise_tv_chambolle_pytorch(model.opt_obja, weight=weight, axis=None, padding=1)
            model.opt_objp.data = denoise_tv_chambolle_pytorch(model.opt_objp, weight=weight, axis=None, padding=1)
            vprint(f"Apply TV denoising with weight = {weight} on obja and objp at iter {niter}", verbose=self.verbose)
    
    def apply_tv_denoise(self, model, niter):
        ''' Apply total-variation denoising on object using PyTorch's TV denoising implementation '''
        # Note that the `object_denoise_tv_pytorch` is applied on stacked 2D object, so it's applying on (omode,z,ky,kx)
        tv_denoise_freq = self.constraint_params['tv_denoise']['freq']
        weights         = self.constraint_params['tv_denoise']['weights']
        iterations      = self.constraint_params['tv_denoise']['iterations']
        z_padding      = self.constraint_params['tv_denoise']['z_padding']
        if tv_denoise_freq is not None and niter % tv_denoise_freq == 0:
            model.opt_obja.data = object_denoise_tv_pytorch(model.opt_obja, weights=weights, iterations=iterations, z_padding=z_padding)
            model.opt_objp.data = object_denoise_tv_pytorch(model.opt_objp, weights=weights, iterations=iterations, z_padding=z_padding)
            vprint(f"Apply TV denoising with weight = {weights} and iterations = {iterations} on obja and objp at iter {niter}", verbose=self.verbose)

    def apply_obj_butterworth(self, model, niter):
        ''' Apply Butterworth filter on object '''
        # Note that the `object_butterworth_filter` is applied on stacked 2D object, so it's applying on (omode,z,ky,kx)
        obj_butter_freq = self.constraint_params['obj_butterworth']['freq']
        q_low          = self.constraint_params['obj_butterworth']['q_lowpass']
        q_high         = self.constraint_params['obj_butterworth']['q_highpass']
        order         = self.constraint_params['obj_butterworth']['butterworth_order']
        obj_type      = self.constraint_params['obj_butterworth']['obj_type']
        if obj_butter_freq is not None and niter % obj_butter_freq == 0:
            if obj_type in ['amplitude', 'both']:
                model.opt_obja.data = object_butterworth_constraint_torch(model.opt_obja, q_lowpass=q_low, q_highpass=q_high, butterworth_order=order)
            if obj_type in ['phase', 'both']:
                model.opt_objp.data = object_butterworth_constraint_torch(model.opt_objp, q_lowpass=q_low, q_highpass=q_high, butterworth_order=order)
            vprint(f"Apply Butterworth filter with q_low = {q_low}, q_high = {q_high}, and order = {order} on obja and objp at iter {niter}", verbose=self.verbose)

    def forward(self, model, niter):
        # Apply in-place constraints if niter satisfies the predetermined frequency
        # Note that the if check blocks are included in each apply methods so that it's cleaner, and I can print the info with niter
        
        with torch.no_grad():
            # Probe constraints
            self.apply_ortho_pmode  (model, niter)
            self.apply_probe_mask_k (model, niter)
            self.apply_probe_mask_r (model, niter)
            self.apply_fix_probe_int(model, niter)
            # Object constraints
            # self.apply_tv_denoise   (model, niter)
            # self.apply_tv_denoise_chambolle(model, niter)
            self.apply_obj_butterworth(model, niter)
            self.apply_obj_rblur    (model, niter)
            self.apply_obj_zblur    (model, niter)
            self.apply_kr_filter    (model, niter)
            self.apply_kz_filter    (model, niter)
            self.apply_complex_ratio(model, niter)
            self.apply_mirrored_amp (model, niter)
            self.apply_obja_thresh  (model, niter)
            self.apply_objp_postiv  (model, niter)
            # Local tilt constraint
            self.apply_tilt_smooth  (model, niter)

###### Filter and helper functions for constraints ######
def sort_by_mode_int(modes):
    modes_int =  modes.abs().pow(2).sum(tuple(range(1,modes.ndim))) # Sum every but 1st dimension
    _, indices = torch.sort(modes_int, descending=True)
    modes = modes[indices]
    return modes

def orthogonalize_modes_vec(modes, sort = False):
    ''' orthogonalize the modes using SVD'''
    # Input:
    #   modes: input function with multiple modes
    # Output:
    #   ortho_modes: 
    # Note:
    #   This function is a highly vectorized PyTorch implementation of `ptycho\+core\probe_modes_ortho.m` from PtychoShelves
    #   It's numerically equivalent with the following for-loop version but is ~ 10x faster on small complex64 tensors (10,164,164) 
    #   Most indexings arr converted from Matlab (start from 1) to Python (start from 0)
    #   The expected shape of `modes` input is modified into (pmode, Ny, Nx) to be consistent with ptyrad
    #   If you check the orthoganality of each mode, make sure to change the input into complex128 or to modify the default tolerance of torch.allclose.
    #   Note that Matlab's dot(p2,p1) for complex input would implictly apply with the complex conjugate, 
    #   so Matlab's dot() != torch.dot because torch.dot doesn't automatically apply the complex conjugate.
    #   This is pointed out by @dong-zehao in issue #11.
        
    orig_modes_dtype = modes.dtype
    if orig_modes_dtype != torch.complex64:
        modes = torch.complex(modes, torch.zeros_like(modes))
    input_shape = modes.shape
    modes_reshaped = modes.reshape(input_shape[0], -1) # Reshape modes to have a shape of (Nmode, X*Y)
    A = torch.matmul(modes_reshaped, modes_reshaped.H) # A = M @ M^T.conj() = M @ M^H, H is the conjugate transpose

    if A.device.type == 'mps': # Temporary hack because PyTorch MPS backend doesn't seem to implement linalg.eig yet.
        _, evecs = torch.linalg.eig(A.to('cpu'))
        evecs = evecs.to('mps')
    else:
        _, evecs = torch.linalg.eig(A)
   
    # Matrix-multiplication version (N,N) @ (N,YX) = (N,YX)
    ortho_modes = torch.matmul(evecs.H, modes_reshaped).reshape(input_shape)

    # sort modes by their contribution
    if sort:
        ortho_modes = sort_by_mode_int(ortho_modes)
        
    return ortho_modes.to(orig_modes_dtype)

def kr_filter(obj, radius, width):
    ''' Apply kr_filter using the 2D sigmoid filter '''
    
    # Create the filter function W, note that the W has to be corner-centered
    Ny, Nx = obj.shape[-2:]
    mask = make_sigmoid_mask(min(Ny,Nx), radius, width).to(obj.device)
    W = ifftshift2(interpolate(mask[None,None,], size=(Ny,Nx))).squeeze() # interpolate needs 2 additional dimension (N,C,...) for the input than the output dimension
        
    # Filter the obj with filter function Wa, take the real part because Fourier-filtered obj could contain negative values    
    fobj = torch.real(ifft2(fft2(obj) * W[None,None,])) # Apply fft2/ifft2 for only the r(y,x) dimension so the omode and z would be broadcasted
    
    return fobj

def kz_filter(obj, beta_regularize_layers=1, alpha_gaussian=1, obj_type='phase'):
    ''' Apply kz_filter using the arctan filter '''
    # Note: Calculate force of regularization based on the idea that DoF = resolution^2/lambda
        
    device = obj.device
    
    # Generate 1D grids along each dimension
    Npix = obj.shape[-3:]
    kz = fftfreq(Npix[0]).to(device) 
    ky = fftfreq(Npix[1]).to(device) 
    kx = fftfreq(Npix[2]).to(device) 
    
    # Generate 3D coordinate grid using meshgrid
    grid_kz, grid_ky, grid_kx = torch.meshgrid(kz, ky, kx, indexing='ij')

    # Create the filter function Wa. W and Wa is exactly the same as PtychoShelves for now
    W = 1 - torch.atan((beta_regularize_layers * torch.abs(grid_kz) / torch.sqrt(grid_kx**2 + grid_ky**2 + 1e-3))**2) / (torch.pi/2)
    Wa = W * torch.exp(-alpha_gaussian * (grid_kx**2 + grid_ky**2))

    # Filter the obj with filter function Wa, take the real part because Fourier-filtered obj could contain negative values    
    fobj = torch.real(ifftn(fftn(obj, dim=(-3,-2,-1)) * Wa[None,], dim=(-3,-2,-1))) # Apply fftn/ifftn for only spatial dimension so the omode would be broadcasted
    
    if obj_type == 'amplitude':
        fobj = 1+0.9*(fobj-1) # This is essentially a soft obja threshold constraint built into the kz_filter routine for obja
        
    return fobj

def complex_ratio_constraint(model, alpha1, alpha2):
    # https://doi.org/10.1016/j.ultramic.2024.114068
    # https://doi.org/10.1364/OE.18.001981
    # Suggested values for alpha1, alpha2 are 1 and 0
    # For alpha1 = 1, alpha2 = 0, it's suggesting a phase object and phase would not be updated.
    # Namely, objac = exp(-alpha1*Cbar*objp); objpc = objp
    # NOTE that my implementaiton is slightly different from the papers
    # Because for electron ptychography we usually defines a positive phase shift in the transmission function, obj(r) = T(r) = exp(i*sigma*V(r))
    # Hence the object phase = angle(obj) = i*sigma*V(r)
    # So when electron being scattered by the nuclei, it accumulates positive phase shift and slightly less than 1 amplitude
    # Hence the constraint foumula is slightly modified accordingly, so we have positive phase associated with slightly less than 1 amplitude
    
    obja = model.opt_obja
    objp = model.opt_objp
    
    log_obja = torch.log(obja)
    
    # Compute Cbar for the entire object across (omode, z, y, x)
    # Although we can consider repeat this across z slices or omode
    Cbar = (log_obja.abs().sum()) / (objp.abs().sum() + 1e-8)  # Avoid division by zero

    # Compute updated amplitude, note the negative sign for the second term!
    objac = torch.exp((1 - alpha1) * log_obja - alpha1 * Cbar * objp)

    # Compute updated phase, note the negative sign for the second term!
    objpc = (1 - alpha2) * objp - alpha2 / (Cbar + 1e-8) * log_obja # Avoid division by zero
    return objac, objpc, Cbar

import torch
import torch.nn.functional as F
import warnings

def _object_denoise_tv_2p5d_pytorch(current_object: torch.Tensor, weights: list[float], iterations: int, z_padding: int) -> torch.Tensor:
    """
    Performs 2.5D TV denoising in PyTorch on a 3D tensor.

    This function applies a first-order TV penalty (gradient) along the z-axis (dim 0)
    and a second-order TV penalty (Laplacian) along the y and x axes (dims 1, 2).
    It solves:
        argmin_x (0.5 * ||x - x_orig||_2^2 + w_z * ||Grad_z(x)||_1 + w_xy * ||Lap_xy(x)||_1)
    using manual gradient descent steps, making it compatible with `torch.compile`.

    Parameters
    ----------
    current_object : torch.Tensor
        The input 3D tensor to be denoised, shape [depth, height, width].
    weights : list[float]
        A list of two regularization weights: [z_weight, xy_weight].
    iterations : int
        The number of optimization iterations to perform.
    z_padding : int
        Symmetric zero-padding to apply to the z-axis before denoising.

    Returns
    -------
    torch.Tensor
        The denoised 3D tensor.
    """
    # --- 1. Input Validation and Setup ---
    if torch.is_complex(current_object):
        warnings.warn(
            "TV denoising is applied to real-valued tensors. "
            "Returning the original complex tensor without modification.",
            UserWarning,
        )
        return current_object

    if not torch.is_floating_point(current_object):
        current_object = current_object.float()

    device = current_object.device
    z_weight, xy_weight = weights
    
    # --- 2. Apply Z-Padding ---
    if z_padding > 0:
        # PyTorch F.pad works on (..., D, H, W), so we pad the D dimension (dim 0).
        padded_object = F.pad(current_object, (0, 0, 0, 0, z_padding, z_padding), "constant", 0)
    else:
        padded_object = current_object

    denoised_object = padded_object.clone()
    lr = 0.01

    # --- 3. Kernel Definitions ---
    # Kernel for first derivative (gradient) along the z-axis (backward difference)
    z_gradient_kernel = torch.tensor([
        [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]],
        [[0.,0.,0.],[0.,-1.,0.],[0.,0.,0.]],
        [[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]]
    ], device=device).unsqueeze(0).unsqueeze(0)

    # Kernel for second derivative (Laplacian) in the xy-plane
    xy_laplacian_kernel = torch.tensor([
        [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]],
        [[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]],
        [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
    ], device=device).unsqueeze(0).unsqueeze(0)

    # --- 4. Manual Optimization Loop ---
    for i in range(iterations):
        denoised_object.requires_grad_(True)
        fidelity_loss = 0.5 * F.mse_loss(denoised_object, padded_object, reduction='sum')
        total_loss = fidelity_loss
        
        # Reshape for 3D convolution: (N, C, D, H, W)
        obj_for_conv = denoised_object.unsqueeze(0).unsqueeze(0)

        # Z-Gradient Regularization
        if z_weight > 0:
            z_grad_obj = F.conv3d(obj_for_conv, z_gradient_kernel, padding='same')
            total_loss += z_weight * torch.sum(torch.abs(z_grad_obj))

        # XY-Laplacian Regularization
        if xy_weight > 0:
            xy_lap_obj = F.conv3d(obj_for_conv, xy_laplacian_kernel, padding='same')
            total_loss += xy_weight * torch.sum(torch.abs(xy_lap_obj))

        grad = torch.autograd.grad(total_loss, denoised_object, only_inputs=True)[0]

        with torch.no_grad():
            denoised_object -= lr * grad

    # --- 5. Remove Padding and Return ---
    if z_padding > 0:
        # Crop the z-axis (dimension 0)
        final_object = denoised_object[z_padding:-z_padding, :, :]
    else:
        final_object = denoised_object

    return final_object.detach()

def object_denoise_tv_pytorch(current_object: torch.Tensor, weights: list[float], iterations: int, z_padding: int) -> torch.Tensor:
    """
    Performs TV denoising on a 4D tensor by applying 2.5D denoising to each time slice.

    The input tensor is expected to have dimensions (time, depth, y, x).
    This function iterates over the time dimension and applies the 
    `_object_denoise_tv_2p5d_pytorch` function to each 3D slice, effectively
    denoising the z (depth) and y,x dimensions while leaving the time axis untouched.

    Parameters
    ----------
    current_object : torch.Tensor
        The input 4D tensor, shape [time, depth, height, width].
    weights : list[float]
        A list of two regularization weights for the 2.5D function: [z_weight, xy_weight].
    iterations : int
        The number of optimization iterations to perform for each slice.
    z_padding : int
        Symmetric zero-padding for the depth (z) axis.

    Returns
    -------
    torch.Tensor
        The denoised 4D tensor.
    """
    # --- 1. Input Validation ---
    if current_object.dim() != 4:
        raise ValueError(f"Input tensor must be 4D, but got {current_object.dim()}D")

    # --- 2. Iterate over Time Dimension ---
    denoised_slices = []
    for t in range(current_object.shape[0]):
        # Get the 3D slice for the current time step
        slice_3d = current_object[t]
        
        # Apply the 2.5D denoising function to the slice
        denoised_slice = _object_denoise_tv_2p5d_pytorch(
            current_object=slice_3d,
            weights=weights,
            iterations=iterations,
            z_padding=z_padding
        )
        denoised_slices.append(denoised_slice)

    # --- 3. Stack Results ---
    # Combine the list of denoised 3D slices back into a 4D tensor
    return torch.stack(denoised_slices, dim=0)

@torch.compiler.disable
def denoise_tv_chambolle_pytorch(
    current_object: torch.Tensor,
    weight: float,
    axis=None,
    padding: int = None,
    eps: float = 2.0e-4,
    max_num_iter: int = 200,
    scaling=None,
):
    """
    Perform total-variation denoising on n-dimensional images using PyTorch.

    This is a PyTorch implementation of the Rudin, Osher, and Fatemi (ROF)
    model using Chambolle's algorithm.

    Parameters
    ----------
    current_object : torch.Tensor
        The input tensor to be denoised.
    weight : float
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    axis : int or tuple, optional
        Axes along which to perform denoising. If None (default), denoising
        is performed on all axes.
    padding : int, optional
        The size of padding to apply along the denoising axes. This helps
        to reduce boundary artifacts.
    eps : float, optional
        Relative difference of the cost function that determines the stop
        criterion. The algorithm stops when `(E_new - E_old) < eps * E_init`.
    max_num_iter : int, optional
        Maximal number of iterations for the optimization.
    scaling : list or torch.Tensor, optional
        Scale factor for the TV norm on each axis. Must have the same
        length as the number of axes being denoised.

    Returns
    -------
    torch.Tensor
        The denoised tensor.
    """
    # Ensure input is a floating point tensor for calculations
    if not current_object.is_floating_point():
        current_object = current_object.float()

    device = current_object.device
    dtype = current_object.dtype

    # Preserve the sum of the original object for final normalization
    original_sum = torch.sum(current_object)

    # --- Axis Handling ---
    if axis is None:
        denoise_axes = list(range(current_object.dim()))
    elif isinstance(axis, int):
        denoise_axes = [axis]
    else:
        denoise_axes = list(axis)
    
    num_axes = len(denoise_axes)

    # --- Padding ---
    if padding is not None and padding > 0:
        # F.pad expects a flat list for padding: (pad_left_dimN, pad_right_dimN, ...)
        # We construct it by iterating through dimensions in reverse order.
        pad_arg = [0] * (current_object.dim() * 2)
        for ax in denoise_axes:
            # Dims are counted from the end for F.pad's argument
            idx = (current_object.dim() - 1 - ax) * 2
            pad_arg[idx] = padding
            pad_arg[idx + 1] = padding
        padded_object = F.pad(current_object, pad_arg, mode="constant", value=0)
    else:
        padded_object = current_object

    # --- Initialize Tensors ---
    # Dual variable `p` has a dimension for each denoising axis
    p_shape = (num_axes,) + padded_object.shape
    p = torch.zeros(p_shape, dtype=dtype, device=device)
    g = torch.zeros_like(p)
    d = torch.zeros_like(padded_object)
    
    # Use a separate variable for the iteratively updated object
    updated_object = padded_object.clone()

    E_init = 0.0
    E_previous = 0.0

    # --- Main Iteration Loop ---
    for i in range(max_num_iter):
        if i > 0:
            # --- Update `d` (divergence of p) ---
            # This calculates d = -div(p) using backward differences
            d = -torch.sum(p, dim=0)
            for p_idx, img_ax in enumerate(denoise_axes):
                # Create slices for efficient tensor indexing
                d_slice = [slice(None)] * padded_object.dim()
                p_slice = [slice(None)] * p.dim()

                d_slice[img_ax] = slice(1, None)
                p_slice[p_idx + 1] = slice(0, -1)
                p_slice[0] = p_idx
                
                d[tuple(d_slice)] += p[tuple(p_slice)]

            updated_object = padded_object + d
        
        E = torch.sum(d**2)

        # --- Update `g` (gradient of the updated object) ---
        # This calculates the forward-difference gradient
        g.zero_() # Reset gradient
        for p_idx, img_ax in enumerate(denoise_axes):
            diffs = torch.diff(updated_object, dim=img_ax)
            
            # Place differences into the gradient tensor `g`
            g_slice = [slice(None)] * p.dim()
            g_slice[p_idx + 1] = slice(0, -1)
            g_slice[0] = p_idx
            
            g[tuple(g_slice)] = diffs
        
        if scaling is not None:
            if not isinstance(scaling, torch.Tensor):
                scaling = torch.tensor(scaling, dtype=dtype, device=device)
            scaling = scaling / torch.max(scaling)
            # Reshape for broadcasting: e.g., (num_axes, 1, 1) for a 2D image
            g *= scaling.view(-1, *([1] * updated_object.dim()))

        # --- Update dual variable `p` ---
        norm = torch.sqrt(torch.sum(g**2, dim=0, keepdim=True))
        E += weight * torch.sum(norm)
        
        tau = 1.0 / (2.0 * num_axes)
        
        denom = 1.0 + (tau / weight) * norm
        
        p = (p - tau * g) / denom

        # --- Check for Convergence ---
        E_current = E / padded_object.numel()
        if i == 0:
            E_init = E_current
        else:
            if torch.abs(E_previous - E_current) < eps * E_init:
                break
        
        E_previous = E_current

    # --- Un-padding ---
    if padding is not None and padding > 0:
        unpad_slices = [slice(None)] * updated_object.dim()
        for ax in denoise_axes:
            unpad_slices[ax] = slice(padding, -padding)
        updated_object = updated_object[tuple(unpad_slices)]

    # --- Final Normalization ---
    # Rescale to preserve the original object's total sum/energy
    final_sum = torch.sum(updated_object)
    if final_sum > 1e-9: # Avoid division by zero
        updated_object = updated_object / final_sum * original_sum

    return updated_object

import torch

def _object_butterworth_constraint_torch(
    current_object, q_lowpass, q_highpass, butterworth_order
):
    """
    Butterworth filter implemented in PyTorch.

    Parameters
    ----------
    current_object: torch.Tensor
        Current object estimate.
    q_lowpass: float
        Cut-off frequency in A^-1 for low-pass butterworth filter.
    q_highpass: float
        Cut-off frequency in A^-1 for high-pass butterworth filter.
    butterworth_order: float
        Butterworth filter order. Smaller values result in a smoother filter.
        
    Returns
    -------
    constrained_object: torch.Tensor
        Constrained object estimate.
    """
    # Ensure all calculations are done on the same device as the input tensor
    device = current_object.device

    # Create frequency coordinates for each dimension
    qz = torch.fft.fftfreq(current_object.shape[0], 1., device=device)
    qx = torch.fft.fftfreq(current_object.shape[1], 1., device=device)
    qy = torch.fft.fftfreq(current_object.shape[2], 1., device=device)

    # Create a 3D grid of frequency magnitudes
    qza, qxa, qya = torch.meshgrid(qz, qx, qy, indexing="ij")
    qra = torch.sqrt(qza**2 + qxa**2 + qya**2)

    # Initialize the filter envelope
    env = torch.ones_like(qra)

    # Apply high-pass filter if specified
    if q_highpass:
        env *= 1 - 1 / (1 + (qra / q_highpass) ** (2 * butterworth_order))
    
    # Apply low-pass filter if specified
    if q_lowpass:
        env *= 1 / (1 + (qra / q_lowpass) ** (2 * butterworth_order))

    # Apply the filter in Fourier space, preserving the mean of the original object
    current_object_mean = torch.mean(current_object)
    current_object = current_object - current_object_mean
    
    # Forward FFT, apply filter, and inverse FFT
    current_object_fft = torch.fft.fftn(current_object)
    filtered_object_fft = current_object_fft * env
    current_object = torch.fft.ifftn(filtered_object_fft)
    current_object = current_object + current_object_mean

    return current_object.real

def object_butterworth_constraint_torch(
    current_object, q_lowpass, q_highpass, butterworth_order
):
    """
    Wrapper function for the Butterworth filter for 4D object.

    Parameters
    ----------
    current_object: torch.Tensor
        Current object estimate.
    q_lowpass: float
        Cut-off frequency in A^-1 for low-pass butterworth filter.
    q_highpass: float
        Cut-off frequency in A^-1 for high-pass butterworth filter.
    butterworth_order: float
        Butterworth filter order. Smaller values result in a smoother filter.

    Returns
    -------
    constrained_object: torch.Tensor
        Constrained object estimate.
    """
    
    # Check if the input is a 4D tensor
    if current_object.dim() != 4:
        raise ValueError(f"Input tensor must be 4D, but got {current_object.dim()}D")

    # Apply the Butterworth filter to each time slice
    denoised_slices = []
    for t in range(current_object.shape[0]):
        slice_3d = current_object[t]
        denoised_slice = _object_butterworth_constraint_torch(
            slice_3d, q_lowpass, q_highpass, butterworth_order
        )
        denoised_slices.append(denoised_slice)

    # Stack the denoised slices back into a 4D tensor
    return torch.stack(denoised_slices, dim=0)
    
    