import jax.numpy as jnp
import cato.cato as ct
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import optax

df = pd.read_csv("text_file_sed_w34.ecsv", comment="#", delim_whitespace=True, names=["name","energy","flux","err_low","err_high","is_ul"])

# Function to extract the SED from clouds
def extract_dataset(df, cname):
    sub    = df[df["name"] == cname]
    energy = sub["energy"].values
    flux   = sub["flux"].values
    err_l  = sub["err_low"].values
    err_h  = sub["err_high"].values
    is_ul  = sub["is_ul"].values

    det = (is_ul == 0)
    Eg_data      = jnp.array(energy[det])
    flux_data    = jnp.array(flux[det])
    flux_data_err= jnp.array(err_h[det])

    ul = (is_ul == 1)
    Eg_ul   = jnp.array(energy[ul])
    flux_ul = jnp.array(flux[ul])

    return Eg_data, flux_data, flux_data_err, Eg_ul, flux_ul

# Physical constants and the pion production cross-section
mp=938.272e6     # eV
mpCGS=1.6726e-24 # g 
Tp_arr=np.logspace(7,15,801)
Eg_arr=np.logspace(8,15,701) # Tp_arr
RadXS=ct.Cross_Section_Rad(Tp_arr, Eg_arr)
d_sigma_g=RadXS.func_d_sigma_g()
eps_nucl=RadXS.func_enhancement()

def func_G0(r, Tp, pars, theta):

    alpha, tage = pars
    D0=(10.0**theta[1])*1.1e28
    xiSNR=10.0**theta[2]

    def func_Q0(Tp, xiSNR, alpha):

        xmin=jnp.sqrt((1.0e8+mp)**2-mp**2)/mp
        xmax=jnp.sqrt((1.0e15+mp)**2-mp**2)/mp
        x=jnp.logspace(jnp.log10(xmin),jnp.log10(xmax),5000)
        Gam=jnp.trapezoid(x**(2.0-alpha)*(jnp.sqrt(x**2+1.0)-1.0),x)

        p=jnp.sqrt((Tp+mp)**2-mp**2) # eV
        vp=p/(Tp+mp)

        ESNR=1.0e51*6.242e+11 # eV -> Average kinetic energy of SNRs
        QE=(xiSNR*ESNR/(mp**2*vp*Gam))*(p/mp)**(2.0-alpha)

        return QE # eV^-1

    p=jnp.sqrt((Tp+mp)**2-mp**2) # eV
    vp=p/(Tp+mp)
    # Diff=D0*vp*(p/1.0e9)**delta # cm^2/s
    pb=312.0e9
    Diff=D0*(365.0*86400.0/(3.08567758e18)**2)*vp*(p/1.0e9)**0.63/(1.0+(p/pb)**2)**0.1 # pc^2/yr
    Diff*=3.08567758e18**2/(365.0*86400.0) # cm^2/s
    Q0=func_Q0(Tp, xiSNR, alpha) # eV^-1
    G0=Q0/((4.0*jnp.pi*Diff*tage))**1.5
    G0*=jnp.exp(-r**2/(4.0*Diff*tage))

    return G0*vp*3.0e10/(4.0*jnp.pi) # eV^-1 cm^-2 s^-1 sr^-1

def func_phi(theta, Eg):

    # Fitted
    r_cl=(10.0**theta[0])*3.086e18   # cm -> Distance to cloud

    # Fixed
    alpha=4.4                # -> Injection index
    tage=3.0e4*365.0*86400.0 # s -> Age of SNR
    M_cl=1.0e4*1.989e33      # g -> Mass of cloud
    dSNR=1.6e3*3.086e18      # cm -> Distance to SNR

    pars=jnp.array([alpha, tage])  
    jE=func_G0(r_cl, Tp_arr, pars, theta) # eV^-1 cm^-2 s^-1 sr^-1

    integrand=eps_nucl[:, jnp.newaxis]*4.0*jnp.pi*jE[:, jnp.newaxis]*d_sigma_g
    phi=jnp.trapezoid(integrand, Tp_arr, axis=0)
    phi*=(M_cl/(1.4*mpCGS))/(4.0*jnp.pi*dSNR**2) # eV^-1 cm^-2 s^-1 

    phi_pred=Eg**2*jnp.interp(Eg*1.0e6, Eg_arr, phi)*1.0e6

    return phi_pred # MeV cm^-2 s^-1 

def loss_fn(params, datasets):
    loss=0.0
    for i, (Eg_data, flux_data, err_data, Eg_ul, flux_ul) in enumerate(datasets):
        phi_pred=func_phi(jnp.array([params["log_r_cl"][i], params["log_D0"], params["log_xiSNR"][i]]), Eg_data)
        loss+=jnp.mean(((phi_pred - flux_data) / err_data)**2)

    return loss/len(datasets)

grad_fn=jax.grad(loss_fn)

cnames=["W3_1bis_up", "W3_1bis_down", "W3_2bis", "W3_3bis"] #, "W3_4"]
datasets=[extract_dataset(df, cname) for cname in cnames]

theta={
    "log_r_cl": jnp.array([1.69101628, 1.5996456 , 1.54321328, 1.34126979]),
    "log_D0": -1.16672363,
    "log_xiSNR": jnp.array([-0.61256115, -0.47478913, -0.42637293, -1.023072])
}

# lr=1e-1                        
# n_steps=10

# optimizer=optax.adam(learning_rate=lr)
# opt_state=optimizer.init(theta)

# for step in range(n_steps):
#     grads = grad_fn(theta, datasets)
#     updates, opt_state = optimizer.update(grads, opt_state)
#     theta = optax.apply_updates(theta, updates)
#     if step % 100 == 0:
#         loss = loss_fn(theta, datasets)
#         print(f"step {step:4d}: theta = {theta}, loss = {loss:.6e}")

D0 = (10 ** theta["log_D0"]) * 1.1e28 # cm^2/s
r_cl_vals = (10 ** theta["log_r_cl"]) # pc
xiSNR_vals = 10 ** theta["log_xiSNR"]/0.15
Mcl_Dame = np.array([0.918, 0.912, 1.04, 1.04])

print("\n=== Diffusion coefficient ===")
print(f"D0 = {10.0**theta['log_D0']:.6f} x 1.1e28 = {D0:.3e}  cm^2/s")

print("\n=== Cloud Parameters      ===")
print(f"{'Cloud':<12} | {'r_cl [pc]':>12} | {'Mcl (1e4 Msol)':>15} | {'Mcl(Dame) (1e4 Msol)':>15}")
print("-"*70)

for i in range(len(theta["log_r_cl"])):
    print(f"{cnames[i]:<12} | "
            f"{r_cl_vals[i]:>12.3e} | "
            f"{xiSNR_vals[i]:>15.3e} | "
            f"{Mcl_Dame[i]:>15.3e}"  
        )

print(" ")

fs=18
fig, ax=plt.subplots(figsize=(8, 6))

colors = ['red', 'green', 'blue', 'orange']

for i, (Eg_data, flux_data, err_data, Eg_ul, flux_ul) in enumerate(datasets):
    phi_pred=func_phi(jnp.array([theta["log_r_cl"][i], theta["log_D0"], theta["log_xiSNR"][i]]), Eg_arr*1.0e-6)
    ax.plot(Eg_arr*1.0e-6, phi_pred, '-', color=colors[i])

    ax.errorbar(
        Eg_data, flux_data,
        yerr=err_data,
        fmt="o", color=colors[i], label=cnames[i]
    )

    ax.errorbar(
        Eg_ul, flux_ul,
        yerr=0.15*flux_ul,
        uplims=True,
        fmt="s", color=colors[i]
    )

ax.set_xlim(1.0e2, 1.0e5)
ax.set_ylim(1.0e-7, 2.0e-5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r'$E_\gamma {\rm (MeV)}$', fontsize=fs)
ax.set_ylabel(r'$E_\gamma^2 \, \phi {\rm (MeV\, s^{-1}\, sr^{-1})}$', fontsize=fs)

ax.tick_params(axis='x', labelsize=fs)
ax.tick_params(axis='y', labelsize=fs)

ax.grid(ls="--", alpha=0.5)
ax.legend(fontsize=fs, ncol=1, loc='upper right')
plt.tight_layout()
plt.savefig("fg_SED.png")