import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

# Getting poisson noise
def num_emitted_photons(rate, t_exp):
    return rate * t_exp

# Getting Sky background noise
def sky_noise(sky_count_rate, t_exp, psf_fwhm):
    # Calculate the image area in arcsec^2
    Image_area = psf_fwhm**2 * np.pi/4

    return sky_count_rate * Image_area * t_exp


def get_dark_current(dark_current_rate, num_pixels, t_exp):
    return dark_current_rate.to(1/u.s) * num_pixels * t_exp

# Getting read noise
def get_read_noise(read_noise, num_pixels):
    return (read_noise)**2 * num_pixels

# Get the number of pixels from focal plane scale,psf_fwhm and pixel size
def get_pix_num(focal_plane_scale, psf_fwhm, pixel_size):
    image_area = (psf_fwhm/focal_plane_scale)**2 * np.pi/4
    return image_area / ((pixel_size.to(u.mm))**2)

def get_sky_background_uncertainty(sky_count_rate, 
                                   t_exp, 
                                   psf_fwhm, 
                                   sky_background_estimation_psf_fwhm,
                                   focal_plane_scale,
                                   pixel_size,):
    # Get sky_count_rate_uncertainty
    num_pixels =np.max([get_pix_num(focal_plane_scale, sky_background_estimation_psf_fwhm, pixel_size).value,1]) # At least 1 pixel
    sky_count_rate_uncertainty = sky_count_rate / np.sqrt(num_pixels)
    
    sky_bkg_unc = sky_noise(sky_count_rate_uncertainty, t_exp, psf_fwhm)

    return sky_bkg_unc

def signal_to_noise(source_count_rate, 
                    sky_count_rate, 
                    read_noise,
                    dark_current_rate,
                     t_exp, 
                     num_pixels, 
                     psf_fwhm,
                     num_coadds=1,
                     add_sky_background_uncertainty=None,
                     debug=False,
                     ):
    
    bkg = 0
    # Calculate the total signal
    total_signal = num_emitted_photons(source_count_rate,t_exp)

    poisson_noise = np.sqrt(total_signal)

    sky_noise_rate = sky_noise(sky_count_rate, t_exp, psf_fwhm)

    dark_current_noise = get_dark_current(dark_current_rate, num_pixels, t_exp)

    read_noise_rate = get_read_noise(read_noise, num_pixels)

    if add_sky_background_uncertainty is not None:
        sky_bkg_unc = get_sky_background_uncertainty(sky_count_rate, 
                                                      t_exp, 
                                                      psf_fwhm, 
                                                      add_sky_background_uncertainty["sky_background_estimation_psf_fwhm"],
                                                      add_sky_background_uncertainty["focal_plane_scale"],
                                                      add_sky_background_uncertainty["pixel_size"],)
        # Calculate the total noise
        total_noise = np.sqrt(total_signal + sky_noise_rate + read_noise_rate + dark_current_noise + sky_bkg_unc)
        bkg = 1

    else:
        # Calculate the total noise
        total_noise = np.sqrt(total_signal + sky_noise_rate + read_noise_rate + dark_current_noise)

    snr = total_signal / total_noise

    fractional_poisson_noise = poisson_noise / total_noise
    fractional_sky_noise = np.sqrt(sky_noise_rate) / total_noise
    fractiona_dark_current = np.sqrt(dark_current_noise) / total_noise
    fractional_read_noise = np.sqrt(read_noise_rate) / total_noise

    snr_dict = {
        'total_signal': total_signal,
        'poisson_noise': poisson_noise,
        'sky_noise_rate': sky_noise_rate,
        'dark_current': dark_current_noise,
        'read_noise_rate': read_noise_rate,
        'total_noise': total_noise,
        'fractional_poisson_noise': fractional_poisson_noise,
        'fractional_sky_noise': fractional_sky_noise,
        'fractional_dark_current': fractiona_dark_current,
        'fractional_read_noise': fractional_read_noise,
        'snr': np.sqrt(num_coadds)*snr
    }
    if bkg == 1:
        fractional_bkg_noise_uncertainty = np.sqrt(sky_bkg_unc) / total_noise
        snr_dict['fractional_bkg_noise_uncertainty'] = fractional_bkg_noise_uncertainty
    if debug:
        print(snr_dict)

    return snr_dict

def plot_noise_rates_vs_t_exps(t_exps,
                                noise_simulations_dict,
                               logy_scale=False,
                               ssize=5,
                               ax=None):
    
    fractional_sky_noise_l = noise_simulations_dict['fractional_sky_noise_l']
    fractional_poisson_noise_l = noise_simulations_dict['fractional_poisson_noise_l']
    fractional_read_noise_l = noise_simulations_dict['fractional_read_noise_l']
    fractional_dark_current_noise_l = noise_simulations_dict['fractional_dark_current_noise_l']

    # Create a new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Customize the lines with different styles and colors
    ax.scatter(t_exps, fractional_sky_noise_l, label='Sky Noise Rate', color='blue', s=ssize, marker='o')
    ax.scatter(t_exps, fractional_poisson_noise_l, label='Poisson Noise Rate', color='green',s=ssize, marker='o')
    ax.scatter(t_exps, fractional_read_noise_l, label='Read Noise Rate', color='red', s=ssize, marker='o')
    ax.scatter(t_exps, fractional_dark_current_noise_l, label='Dark Current Noise Rate', color='orange', s=ssize, marker='o')

    # Add titles and labels
    ax.set_title('Noise Rates vs Exposure Time', fontsize=16)
    ax.set_xlabel('Exposure Time (s)', fontsize=14)
    ax.set_ylabel('Fractional Noise Rate Contributions', fontsize=14)

    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.2)

    # Customize the legend
    ax.legend(loc='best', fontsize=12)

    if logy_scale:
        ax.set_yscale('log')

    # Show the plot if a new figure was created   
    if ax is None:
        plt.show()


    return ax


def plot_noise_rates_vs_ap_sizes(ap_sizes,
                                noise_simulations_dict,
                               logy_scale=False,
                               ssize=5,
                               optimal_ap_size=None,
                               ax=None):
    
    fractional_sky_noise_l = noise_simulations_dict['fractional_sky_noise_l']
    fractional_poisson_noise_l = noise_simulations_dict['fractional_poisson_noise_l']
    fractional_read_noise_l = noise_simulations_dict['fractional_read_noise_l']
    fractional_dark_current_noise_l = noise_simulations_dict['fractional_dark_current_noise_l']
    fractional_bkg_noise_uncertainty_l = noise_simulations_dict['fractional_bkg_noise_uncertainty_l']
    # Create a new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Customize the lines with different styles and colors
    ax.scatter(ap_sizes, fractional_sky_noise_l, label='Sky Noise Rate', color='blue', s=ssize, marker='o')
    ax.scatter(ap_sizes, fractional_poisson_noise_l, label='Poisson Noise Rate', color='green',s=ssize, marker='o')
    ax.scatter(ap_sizes, fractional_read_noise_l, label='Read Noise Rate', color='red', s=ssize, marker='o')
    ax.scatter(ap_sizes, fractional_dark_current_noise_l, label='Dark Current Noise Rate', color='orange', s=ssize, marker='o')
    ax.scatter(ap_sizes, fractional_bkg_noise_uncertainty_l, label='Background Noise Uncertainity', color='purple', s=ssize, marker='o')


    # Add a vertical line to indicate the optimal aperture size
    if optimal_ap_size is not None:
        ax.axvline(optimal_ap_size.value, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Add titles and labels
    ax.set_title('Noise Rates vs Background Estimation Arpeture Sizes', fontsize=16)
    ax.set_xlabel(r'Aperture Sizes [arc sec sq.]', fontsize=14)
    ax.set_ylabel('Fractional Noise Rate Contributions', fontsize=14)

    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.2)

    # Customize the legend
    ax.legend(loc='best', fontsize=12)

    if logy_scale:
        ax.set_yscale('log')

    # Show the plot if a new figure was created   
    if ax is None:
        plt.show()


    return ax


def plot_noise_rates_vs_photon_rate(source_count_rate_l,
                                    noise_simulations_dict,
                                    ssize=5,
                                    ):
        # Plot all the values
    plt.figure(figsize=(10, 6))
    fractional_sky_noise_l = noise_simulations_dict['fractional_sky_noise_l']
    fractional_poisson_noise_l = noise_simulations_dict['fractional_poisson_noise_l']
    fractional_read_noise_l = noise_simulations_dict['fractional_read_noise_l']
    fractional_dark_current_noise_l = noise_simulations_dict['fractional_dark_current_noise_l']

    # Customize the lines with different styles and colors
    plt.scatter(source_count_rate_l, fractional_sky_noise_l, label='Sky Noise Rate Fraction', color='blue',s=ssize, marker='o')
    plt.scatter(source_count_rate_l, fractional_poisson_noise_l, label='Poisson Noise Rate Fraction', color='green',s=ssize, marker='o')
    plt.scatter(source_count_rate_l, fractional_read_noise_l, label='Read Noise Rate Fraction', color='red', s=ssize, marker='o')
    plt.scatter(source_count_rate_l, fractional_dark_current_noise_l, label='Dark Current Noise Rate Fraction', color='orange', s=ssize, marker='o')


    # Add titles and labels
    plt.title('Noise Rates vs Source Count Rate', fontsize=16)
    plt.xlabel('Source Count Rate (photon/s)', fontsize=14)
    plt.ylabel('Fractional Noise Rate Contributions', fontsize=14)

    # Add grid lines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.2)

    # Customize the legend
    plt.legend(loc='best', fontsize=12)

    # Show the plot
    plt.show()


def plot_snr_vs_t_exps(t_exps, 
                       snr_l,
                       ssize=5,
                       lbl:str='SNR',
                       ylbl:str='SNR',
                       ax=None,
                       ax_color='cyan',
                       set_logyscale:bool=False):

    if ax is None:
        fig,ax = plt.subplots(figsize = (10,6))

    else:
        ax=ax

    # Customize the lines with different styles and colors
    ax.scatter(t_exps, snr_l, label=lbl, color=ax_color, marker='o',s=ssize)

    # Add titles and labels
    ax.set_title('SNR vs Exposure Time', fontsize=16)
    ax.set_xlabel('Exposure Time (s)', fontsize=14)
    ax.set_ylabel(ylbl, fontsize=14)

    if set_logyscale:
        ax.set_yscale('log')

    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.2)

    # Customize the legend
    plt.legend(loc='best', fontsize=12)

    if ax is None:
        plt.show()

    return ax

def noise_simulations(params, 
                       t_exps=None,
                       source_count_rate_l=None,
                       add_sky_background_uncertainty_l=None,
                       varying_qty:str='time',
                       num_coadds=1,
                       ):
    snr_l = []
    sky_noise_rate_l = []
    total_signal_l = []
    read_noise_rate_l = []
    fractional_poisson_noise_l = []
    fractional_sky_noise_l = []
    fractional_dark_current_noise_l = []
    fractional_read_noise_l = []
    fractional_bkg_noise_uncertainty_l = []

    bkg = 0

    if varying_qty == 'time':
        values = t_exps
        key = 't_exp'
    elif varying_qty == 'source_rate':
        values = source_count_rate_l
        key = 'source_count_rate'

    elif varying_qty == 'aperture_size':
        values = add_sky_background_uncertainty_l
        key = 'add_sky_background_uncertainty'
        bkg = 1
    else:
        raise ValueError("varying_qty must be either 'time' or 'source_rate'")

    for value in values:
        snr = signal_to_noise(**params, **{key: value}, num_coadds=num_coadds)
        snr_l.append(snr['snr'])
        sky_noise_rate_l.append(snr['sky_noise_rate'])
        total_signal_l.append(snr['total_signal'])
        read_noise_rate_l.append(snr['read_noise_rate'])
        fractional_poisson_noise_l.append(snr['fractional_poisson_noise'])
        fractional_sky_noise_l.append(snr['fractional_sky_noise'])
        fractional_dark_current_noise_l.append(snr['fractional_dark_current'])
        fractional_read_noise_l.append(snr['fractional_read_noise'])
        if bkg == 1:
            fractional_bkg_noise_uncertainty_l.append(snr['fractional_bkg_noise_uncertainty'])

    # if bkg == 0:
    #     return snr_l, sky_noise_rate_l, total_signal_l, read_noise_rate_l, fractional_poisson_noise_l, fractional_sky_noise_l, fractional_read_noise_l, fractional_dark_current_noise_l
    
    # elif bkg == 1:
    #     return snr_l, sky_noise_rate_l, total_signal_l, read_noise_rate_l, fractional_poisson_noise_l, fractional_sky_noise_l, fractional_read_noise_l, fractional_dark_current_noise_l,fractional_bkg_noise_uncertainty_l
            
    noise_simulations_dict = {
        'snr_l': snr_l,
        'sky_noise_rate_l': sky_noise_rate_l,
        'total_signal_l': total_signal_l,
        'read_noise_rate_l': read_noise_rate_l,
        'fractional_poisson_noise_l': fractional_poisson_noise_l,
        'fractional_sky_noise_l': fractional_sky_noise_l,
        'fractional_dark_current_noise_l': fractional_dark_current_noise_l,
        'fractional_read_noise_l': fractional_read_noise_l,
    }

    if bkg == 1:
        noise_simulations_dict['fractional_bkg_noise_uncertainty_l'] = fractional_bkg_noise_uncertainty_l


    return noise_simulations_dict