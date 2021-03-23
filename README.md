# SSVEP-Impulse-Response

Some demos about how to decompose and reconstruct an SSVEP based on the superposition model, such as in the following literatures:

A. Capilla, P. Pazo-Alvarez, A. Darriba, P. Campo, and J. Gross, ''Steady-state visual evoked potentials can be explained by temporal superposition of transient event-related responses,'' PLoS ONE, vol. 6, no. 1, Jan. 2011, Art. no. e14543. 860

J. Thielen, P. van den Broek, J. Farquhar, and P. Desain, ''Broad-Band visually evoked potentials: Re (Con) volution in brain-computer interfacing,'' PLoS ONE, vol. 10, no. 7, 2015, Art. no. e0133797. 

S. Nagel and M. Spüler, ''Modelling the brain response to arbitrary visual stimulation patterns for a flexible high-speed brain-computer interface,'' PLoS ONE, vol. 13, no. 10, Oct. 2018, Art. no. e0206107.

In the literature, some studies reported that the evoked response can be recognized as the response to a single impulse, i.e., impulse response. Furthermore, the steady-state responses or the code-modulated responses could be considered as the temporal superposition of the impulse responses. 

Based on this idea, we may  decompose an SSVEP into two parts: 1) the impulse response and 2) the periodic impulse (i.e., visual stimulus).

# Demos
Here we provide four demos to show how to decompose an SSVEP and reconstruct a new SSVEP. Then we measure the difference between the original SSVEP and the reconstructed SSVEP by using the correlation and mean square error (MSE)

Demo1: we decompose an SSVEP and then reconstruct it (corresponding to the same stimulus). We measure their difference by using the correlation and MSE. 

Demo2: we decompose an SSVEP corresponding to the i-th stimulus and then reconstruct a new SSVEP corresponding to the (i+1)-th stimulus). We measure the difference between the original SSVEP and the reconstructed SSVEP (both corresponding to the (i+1)-th stimulus.

Demo1a: Demo 1a is similar to Demo1. But this demo uses the canonical correlation analysis (rather than ALS) to find the spatial filter and impulse response when we decompose the SSVEP.

Demo2a: Demo 1a is similar to Demo2. But this demo uses the canonical correlation analysis (rather than ALS) to find the spatial filter and impulse response when we decompose the SSVEP.

Date: 23 March 2021

All of the codes are prepared by Chi Man Wong. If you have any feedback or questions, please send email to chiman465@gmail.com.

If you use this code for a publication, please cite the following paper:

@article{wong2021transferring,
   
   title={Transferring Subject-Specific Knowledge Across Stimulus Frequencies in SSVEP-Based BCIs},
   
   author={Wong, Chi Man and Wang, Ze and Rosa, Agostinho C and Chen, CL Philip and Jung, Tzyy-Ping and Hu, Yong and Wan, Feng},
   
   journal={IEEE Transactions on Automation Science and Engineering},
   
   year={2021},
  
  publisher={IEEE}

}
