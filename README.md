## Gamma-ray emissions from clouds around a supernova remnant

We provide a simple model for hadronic gamma-ray emissions from clouds around a supernova remnant. Two ingredients are: i) the solution of the cosmic-ray transport equation from a point source and ii) the differential cross-sections for gamma-ray production.

**Cosmic-ray distribution**: We use the point source solution for the supernova remnant of the following form

$$ 
  f(E)=\frac{Q(E)}{\left[4\pi D(E) t_{\rm age}\right]^{3/2}}\exp\left[-\frac{r_{\rm cl}^2}{4D(E)t_{\rm age}}\right],
$$

where 

$$
  Q(E)=\frac{\xi_{\rm SNR}E_{\rm SNR}}{m_{\rm p}^2c^4\beta\Lambda}\left(\frac{p}{m_{\rm p}c}\right)^{2-\alpha},
  D(E)=\chi D_0 \beta \frac{\left(\frac{\mathcal{R}}{1\\, {\rm GV}}\right)^{0.63}}{\left[1+\left(\frac{\mathcal{R}}{312\\, {\rm GV}}\right)^2\right]^{0.1}}.
$$

We have adopted the form of the injection spectrum as a power-law in momentum and the diffusion coefficient following the form of the average diffusion coefficient fitted from Galactic transport model (see e.g. [Evoli et al. 2019](https://ui.adsabs.harvard.edu/abs/2019PhRvD..99j3023E/abstract)) but with a suppression factor $\chi$. 


**Gamma-ray production cross-section**: We use the library [cato](https://github.com/vhmphan/cato) with the Geant 4 parametrization of the differential cross sections from [Kafexhiu et al. 2014](https://ui.adsabs.harvard.edu/abs/2014PhRvD..90l3014K/abstract). The gamma-ray spectrum from a cloud can be estimated as follows

$$
  \phi(E_\gamma)=\frac{M_{\rm cl}}{4\pi d_{\rm SNR}^2 1.4m_{\rm p}}\int {\rm d}E \\,f(E) v_{\rm p}\varepsilon(E)\frac{{\rm d} \sigma_{pp}(E,E_\gamma)}{{\rm d} E_\gamma}
$$

**Fixed and fitted parameters**: We will fix $E_{\rm SNR}=10^{51}$ erg, $\xi_{\rm SNR}=0.15$, $t_{\rm age}=3\times 10^4$ yr, $d_{\rm SNR}=1.6$ kpc, and $\alpha=4.4$. The fitted parameters are distance from the cloud to the remnant center $r_{\rm cl}$, the cloud mass $M_{\rm cl}$, and the suppression factor for the diffusion coefficient $\chi$.     

**Instruction to use the code**: 
Clone the repository and put the sed file 'text_file_sed_w34.ecsv' in the same folder. If you don't have jax, optax and cato, please install them as follows
```sh
  pip3 install jax
  pip3 install optax
  pip3 install git+https://github.com/vhmphan/cato.git
```
The current version has already the bes fit parameters for the clouds of HB3. If you want to perform the fit again, then uncomment the loops. You can run the code as below  
```sh
  python3 model_ps.py
```

You will the fitted spectra as shown below.

![Fitted spectra from clouds around HB3](https://drive.google.com/uc?export=view&id=1mhFqhmOa1zLAPYApRP3HJz-K-awhpfdu)


