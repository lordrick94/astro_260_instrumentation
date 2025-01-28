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

def signal_to_noise(source_count_rate, 
                    sky_count_rate, 
                    read_noise,
                    dark_current_rate,
                     t_exp, 
                     num_pixels, 
                     psf_fwhm,
                     num_coadds=1,
                     debug=False,
                     ):
    # Calculate the total signal
    total_signal = num_emitted_photons(source_count_rate,t_exp)

    poisson_noise = np.sqrt(total_signal)

    sky_noise_rate = sky_noise(sky_count_rate, t_exp, psf_fwhm)

    dark_current_noise = get_dark_current(dark_current_rate, num_pixels, t_exp)

    read_noise_rate = get_read_noise(read_noise, num_pixels)

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
    if debug:
        print(snr_dict)

    return snr_dict

def plot_noise_rates_vs_t_exps(t_exps,
                               fractional_sky_noise_l,
                               fractional_dark_current_noise_l,
                               fractional_poisson_noise_l,
                               fractional_read_noise_l,
                               logy_scale=False,
                               ssize=5,
                               ax=None):
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

def plot_noise_rates_vs_photon_rate(source_count_rate_l,
                                    fractional_sky_noise_l,
                                    fractional_dark_current_noise_l,
                                    fractional_poisson_noise_l,
                                    fractional_read_noise_l,
                                    ssize=5,
                                    ):
        # Plot all the values
    plt.figure(figsize=(10, 6))

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

    if varying_qty == 'time':
        values = t_exps
        key = 't_exp'
    elif varying_qty == 'source_rate':
        values = source_count_rate_l
        key = 'source_count_rate'
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

    return snr_l, sky_noise_rate_l, total_signal_l, read_noise_rate_l, fractional_poisson_noise_l, fractional_sky_noise_l, fractional_read_noise_l, fractional_dark_current_noise_l