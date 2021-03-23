%% Demo 2 of Decomposition and Reconstruction of SSVEP %%%
% In this code, we provide an example about how to decompose and
% reconstruct an SSVEP based on the superposition model as mentioned in some previous studies, such as 

% A. Capilla, P. Pazo-Alvarez, A. Darriba, P. Campo, and J. Gross, 
% ''Steady-state visual evoked potentials can be explained by temporal 
% superposition of transient event-related responses,'' PLoS ONE, vol. 6, 
% no. 1, Jan. 2011, Art. no. e14543. 

% J. Thielen, P. van den Broek, J. Farquhar, and P. Desain, ''Broad- 
% Band visually evoked potentials: Re (Con) volution in brain-computer 
% interfacing,'' PLoS ONE, vol. 10, no. 7, 2015, Art. no. e0133797. 

% S. Nagel and M. Sp?ler, ''Modelling the brain response to arbitrary visual 
% stimulation patterns for a flexible high-speed brain-computer interface,'' 
% PLoS ONE, vol. 13, no. 10, Oct. 2018, Art. no. e0206107.

% In this demo, we decompose an SSVEP corresponding to the i-th stimulus and then reconstruct a new SSVEP corresponding to the (i+1)-th stimulus. We measure
% the difference between the real SSVEP and the reconstructed SSVEP (both corresponding to the (i+1)-th stimulus) by using the correlation and MSE
%

% This code is prepared by Chi Man Wong (chiman465@gmail.com)
% Date: 23 March 2021
% if you use this code for a publication, please cite the following paper:

% @article{wong2021transferring,
%   title={Transferring Subject-Specific Knowledge Across Stimulus Frequencies in SSVEP-Based BCIs},
%   author={Wong, Chi Man and Wang, Ze and Rosa, Agostinho C and Chen, CL Philip and Jung, Tzyy-Ping and Hu, Yong and Wan, Feng},
%   journal={IEEE Transactions on Automation Science and Engineering},
%   year={2021},
%   publisher={IEEE}
% }

clear all;
close all;
% Please download the SSVEP benchmark dataset for this code
% Wang, Y., et al. (2016). A benchmark dataset for SSVEP-based brain¡Vcomputer interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(10), 1746-1752.
% Then indicate where the directory of the dataset is :
% str_dir=cd; % Directory of the SSVEP Dataset (Change it if necessary)
str_dir='..\Tsinghua dataset 2016\';
addpath('..\mytoolbox\');

num_of_subj=35; % Number of subjects (35 if you have the benchmark dataset)

Fs=250; % sample rate
ch_used=[48 54 55 56 57 58 61 62 63]; % Pz, PO5, PO3, POz, PO4, PO6, O1,Oz, O2 (in SSVEP benchmark dataset)
oz_ch=8;
num_of_trials=5;                    % Number of training trials (1<=num_of_trials<=2)
num_of_harmonics=5;                 % for all cca-based methods
num_of_subbands=1;                  % for filter bank analysis
latencyDelay = round(0.12*Fs);      % for excluding latency response
tw=2;                               % time window length
sig_len=tw*Fs;                       % data length


%bandpass filter
for k=1:num_of_subbands
    Wp = [(8*k)/(Fs/2) 90/(Fs/2)];
    Ws = [(8*k-2)/(Fs/2) 100/(Fs/2)];
    [N,Wn] = cheb1ord(Wp,Ws,3,40);
    [subband_signal(k).bpB,subband_signal(k).bpA] = cheby1(N,0.5,Wn);    
end
%notch
Fo = 50;
Q = 35;
BW = (Fo/(Fs/2))/Q;

[notchB,notchA] = iircomb(Fs/Fo,BW,'notch');
seed = RandStream('mt19937ar','Seed','shuffle');
for sn=1:num_of_subj
    tic
%     load(strcat(str_dir,'\','exampleData.mat'));
    load(strcat(str_dir,'\s',num2str(sn),'.mat'));
    
    %  pre-stimulus period: 0.5 sec
    %  latency period: 0.14 sec
    eeg=data(ch_used,floor(0.5*Fs)+1:floor(0.5*Fs+latencyDelay)+sig_len,:,:);
    
    
    [d1_,d2_,d3_,d4_]=size(eeg);
    d1=d3_;d2=d4_;d3=d1_;d4=d2_;
    no_of_class=d1;
    % d1: num of stimuli
    % d2: num of trials
    % d3: num of channels % Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
    % d4: num of sampling points
    for i=1:1:d1
        for j=1:1:d2
            y0=reshape(eeg(:,:,i,j),d3,d4);            
            y = filtfilt(notchB, notchA, y0.'); %notch
            y = y.';
            for sub_band=1:num_of_subbands
                
                for ch_no=1:d3
                    tmp2=filtfilt(subband_signal(sub_band).bpB,subband_signal(sub_band).bpA,y(ch_no,:));
                    y_sb(ch_no,:) = tmp2(latencyDelay+1:latencyDelay+2*Fs);                    
                end
                
                subband_signal(sub_band).SSVEPdata(:,:,j,i)=reshape(y_sb,d3,length(y_sb),1,1);
            end
            
        end
    end
    
    clear eeg    
    
    pha_val=[0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 ...
        0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5]*pi;
    sti_f=[8:0.2:15.8];
    n_sti=length(sti_f);                     % number of stimulus frequencies
    temp=reshape([1:40],8,5);
    temp=temp';
    target_order=temp(:)';
    
    for k=1:num_of_subbands
        subband_signal(k).SSVEPdata=subband_signal(k).SSVEPdata(:,:,:,target_order); % To sort the orders of the data as 8.0, 8.2, 8.4, ..., 15.8 Hz
        for i=1:no_of_class
            subband_signal(k).signal_template(i,:,:)=mean(subband_signal(k).SSVEPdata(:,:,:,i),3);      % Average the SSVEP across trials
        end
    end
    
    
    for k=1:num_of_subbands   
        sub_fig=1;
        for i=1:2:no_of_class                     
            
            
            % Decompose an SSVEP corresponding to the i-th stimulus
            fs=sti_f(i);
            ph=pha_val(i);
            ssvep=squeeze(subband_signal(k).signal_template(i,:,:));                                    % this is the SSVEP we want to decompose
            y0=ssvep;
            y0_oz=ssvep(oz_ch,:);
            freq_period=1.05*(1./sti_f(i));                                                             % the length of impulse response
            t=[0:sig_len-1]/Fs;
            [H0,h0]=my_conv_H(fs,ph,Fs,tw,60,freq_period);                                              % Create the matrix for convolution calculation
            
            x_hat_oz=y0_oz*H0'*inv(H0*H0');                                                             % impulse response (Oz)
                        
            % Alternating Least Square (Find the impulse response and the
            % spatial filter simultaneously
            w0_old=randn(1,length(ch_used));
            x_hat_old=w0_old*y0*H0'*inv(H0*H0');
            e_old=norm(w0_old*y0-x_hat_old*H0);
            my_err=100;
            iter=1;
            while (my_err(iter)>0.0001 && iter<200)
                w0_new=x_hat_old*H0*y0'*inv(y0*y0');
                x_hat_new=w0_new*y0*H0'*inv(H0*H0');
                e_new=norm(w0_new*y0-x_hat_new*H0);
               
                iter=iter+1;
                my_err(iter)=abs(e_old-e_new);
                w0_old=w0_new;
                w0_old=w0_old/std(w0_old);
                x_hat_old=x_hat_new;
                x_hat_old=x_hat_old/std(x_hat_old);
                e_old=e_new;
            end
            x_hat=x_hat_new;
            
            
            cr=corrcoef(x_hat,x_hat_oz);
            if cr(1,2)<0
                x_hat=-x_hat;
                w0_new=-w0_new;
            end
            
            % Reconstruct a new SSVEP corresponding to the (i+1)-th
            % stimulus
            fs1=sti_f(i+1);
            ph1=pha_val(i+1);
            ssvep1=squeeze(subband_signal(k).signal_template(i+1,:,:));                                 % this is the SSVEP we want to reconstruct
            y1=ssvep1;
            y1_oz=ssvep1(oz_ch,:);
            freq_period1=1.05*(1./sti_f(i));                                                            % the length of impulse response
            
            [H1,h1]=my_conv_H(fs1,ph1,Fs,tw,60,freq_period1);                                           % Create the matrix for convolution calculation           
            y1_hat=x_hat*H1;
            %         x_hat=y*H'*inv(H*H');
            %         y_hat=x_hat*Hs;
            %         if ph_str==pi
            y1_hat(:,1:length(find(y1_hat==0)))=0.8*y1_hat(:,Fs+1:Fs+length(find(y1_hat==0)));
            %         else
            %             y_hat(:,1:length(find(y_hat==0)))=y_hat(:,srate-length(find(y_hat==0))+1:srate);
            %         end
            %         y_hat(:,1:length(find(y_hat==0)))=y_hat(:,end-length(find(y_hat==0))+1:end);
            y1_sf=w0_new*ssvep1;
            
            y1_sf=y1_sf-mean(y1_sf);
            y1_sf=y1_sf/std(y1_sf);
            y1_hat=y1_hat-mean(y1_hat);
            y1_hat=y1_hat/std(y1_hat);
            
            if sn==1
                
                figure(1);
                subplot(4,5,sub_fig); 
                plot(x_hat,'r');
                hold on;
                plot(x_hat_oz,'b');
                hold off;
                xlim([0 length(x_hat)]);
                title([num2str(sti_f(i),'%.1f') '(Hz)']);
            
                figure(2);
                subplot(4,5,sub_fig); 
                plot(y1_sf(1:floor(Fs*0.5)),'r');
                hold on;
                plot(y1_hat(1:floor(Fs*0.5)),'b');
                hold off;
                xlim([0 length(y1_hat(1:floor(Fs*0.5)))]);
                title([num2str(sti_f(i+1),'%.1f') '(Hz)']);
                
            end
            
            ymse(sn,sub_fig)=norm(y1_sf-y1_hat)/length(y1_sf);
            r=corrcoef(y1_sf,y1_hat);
            ycor(sn,sub_fig)=r(1,2);
            sub_fig=sub_fig+1;
        end
    end
    toc 
    disp(sn)
end


