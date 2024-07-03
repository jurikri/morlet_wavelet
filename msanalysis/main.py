def msmain(EEGdata_ch_x_time=None, SR=None, finum=50):
    def ms_morlet_wavelet(finum=50, raw_data=None, SR=None, cpus=12):
        # raw_data shape
        # input shape -> ch x time
        import ray
        import numpy as np
        import time
        from tqdm import tqdm
        import math
        
        def matlab_FIRfilter(msdata = None, SR=1000):
            import numpy as np
            import scipy.signal as scipy_signal
            
            def matlab_windows(t='hamming', filtorder=6600):
                if t == 'hamming':
                    m = filtorder+1
                    
                    isOddLength = np.mod(m, 2)
                    if isOddLength: x = np.array(range(0,int((m-1)/2+1))) / (m-1)
                    else: x = np.array(range(0,int((m/2)-1+1))) / (m-1)
                        
                    a = 0.54 # hamming 에서 fix
                    w = a - (1 - a) * np.cos(2 * np.pi * x)
                    
                    if isOddLength:
                        ix = list(range(len(w)))
                        rix = np.sort(ix)[::-1][1:]
                        w = np.concatenate((w, w[rix]), axis=0)    
                    else:
                        ix = list(range(len(w)))
                        rix = np.sort(ix)[::-1]
                        w = np.concatenate((w, w[rix]), axis=0)
                return w
    
            gfiltorder = int(6600 * (SR/1000))
            winArray = matlab_windows(filtorder = gfiltorder)
            # SR = 1000
            cutoffArray = np.array([0.2500, 50.2500]) # 0.5 / 50 기준
            fNyquist = SR/2
            
    
        # b = firws(g.filtorder, cutoffArray / fNyquist, winArray);
    
    
            m = gfiltorder
            f = cutoffArray / fNyquist
            t = winArray
            
            f = f / 2
            w = t
    
            def matlab_fkernel(m,f1,w):
                m1 = np.array(range(int(-m/2),int(m/2)+1))
                b = np.zeros(len(m1)) * np.nan
                
                b[m1==0] = 2*np.pi*f1
                b[m1!=0] = np.sin(2 * np.pi * f1 * m1[m1!=0]) / m1[m1!=0]
                b = b * w
                b = b/np.sum(b)
                return b
    
            def matlab_fspecinv(b):
                b = -b
                b[int((len(b)-1-1)/2+1)] = b[int((len(b)-1-1)/2+1)]+1
                return b
    
            b = matlab_fkernel(m, f[0], w);
    
            if len(f) == 2:
                b = b + matlab_fspecinv(matlab_fkernel(m, f[1], w));
                # plt.plot(b)
                if True:
                    b = matlab_fspecinv(b); # 특정조건문임 일단 그냥 적용
                    # plt.figure()
                    # plt.plot(b)
            
            groupDelay = int((len(b) - 1) / 2)
            dcArray = [1, msdata.shape[1]+1]
    
            iDc = 0 # 유효하지 않은 loop문 var
            
            ziDataDur = np.min([groupDelay, dcArray[iDc + 1] - dcArray[iDc]]);
    
            chaninds = list(range(len(msdata)))
    
            ms1 = msdata[chaninds][:, np.array(np.ones(groupDelay) * dcArray[iDc], dtype=int)-1]
            ms2 = msdata[chaninds, dcArray[iDc]-1:(dcArray[iDc] + ziDataDur - 1)]
            xin =  np.concatenate((ms1,ms2), axis=1)
    
            # print(np.mean(ms1), np.mean(ms2))
            
            # zi = signal.lfilter_zi(b, a)
            y, zi = scipy_signal.lfilter(b, 1, xin, axis=1, zi=np.zeros((len(chaninds),len(b)-1)))
            
            
            nFrames = 1000 # fix
            # [temp, zi] = filter(b, 1, , [], 2);
            
            ms3 = np.array((range((dcArray[iDc] + groupDelay)-1, (dcArray[iDc + 1] - 1), nFrames)))
            ms4 = np.array([dcArray[iDc + 1]])
            blockArray = np.concatenate((ms3, ms4), axis=0);
            
            # ms3.shape
            
            for iBlock in range(len(blockArray)-1):
                # Filter the data
                
                xin = msdata[chaninds, blockArray[iBlock]-1:(blockArray[iBlock + 1] - 1)]
                y, zi = scipy_signal.lfilter(b, 1, xin, axis=1, zi=zi)
                msdata[chaninds, (blockArray[iBlock] - groupDelay) : (blockArray[iBlock + 1] - groupDelay - 1)+1] = y
            
            
            # temp = filter(b, 1, double(EEG.data(chaninds, ones(1, groupDelay) * (dcArray(iDc + 1) - 1))), zi, 2);
            xin = msdata[chaninds][:,np.array(np.ones(groupDelay) * (dcArray[iDc + 1] - 1)-1, dtype=int)]
            # print(np.mean(xin))
            temp, _ = scipy_signal.lfilter(b, 1, xin, axis=1, zi=zi)
            # print(np.mean(b))
            # print(np.mean(temp))
            
            
            xin = temp[:, (-ziDataDur+1-1):];
            # print(xin.shape, np.mean(xin))
            msdata[chaninds, (dcArray[iDc + 1] - ziDataDur)-1:(dcArray[iDc + 1] - 1)] = xin
            
            return msdata
        
        @ray.remote
        def ms_wavelet(xdata=None, SR=None, finum=50): # 확인 후 속도 개선
            import numpy as np
            # xdata = xdatas[i]
            # input shape -> trial x xlen x channel
            # SR = 1000
    
            tn = xdata.shape[0]
            xlen = xdata.shape[1]
            cn = xdata.shape[2]
            
            msout = []
            min_freq =  1;
            max_freq = finum;
            num_frex = finum;
            frex = np.linspace(min_freq,max_freq,num_frex);
            
            range_cycles = [4, 10];
            beta0 = np.log10(range_cycles[0])
            beta1 = np.log10(range_cycles[1])
    
            s = np.logspace(beta0, beta1, num_frex) / (2*np.pi*frex)
            wavtime = np.arange(-2,2,1/SR)
            half_wave = int((len(wavtime))/2)
            
            nWave = len(wavtime)
            nData = xlen
            nConv = nWave + nData - 1
                
            for channel2use in range(cn):
                tf = np.zeros((len(frex), xlen, tn, 2)) * np.nan;
                
                for fi in range(len(frex)):
                    # create wavelet and get its FFT
                    # the wavelet doesn't change on each trial...
                    wavelet  = np.exp(2*1j*np.pi*frex[fi] * wavtime) * np.exp(-wavtime**2 / (2*s[fi]**2));
    
                    waveletX = np.fft.fft(wavelet,n=nConv);
                    waveletX = waveletX / max(waveletX);
    
                    for trial_i in range(tn):
                        dataX = np.fft.fft(xdata[trial_i, :, channel2use], n=nConv)
                        ms_as = np.fft.ifft(waveletX * dataX);
                        ms_as = ms_as[half_wave+1:-half_wave+2];
                        tf[fi,:,trial_i,0] = np.square(np.abs(ms_as)); # amplitube
                        tf[fi,:,trial_i,1] = ms_as.imag # phase
                        # tf[fi,:,trial_i,2] = ms_as.real
          
                msout.append(tf)
            msout = np.array(msout)
            return msout
        
        raw_data2_storage = np.array(raw_data)
        binlen = 10*SR
        
        if raw_data.shape[1] < binlen: # opendata2에서 발생한 이슈; opendata 는 데이터가 분절되어있어서 6초임
            binlen = raw_data.shape[1]
        
        msbins = np.arange(0, raw_data2_storage.shape[1]-binlen+1, binlen/2, dtype=int)
        result_concat = []
        start = time.time()
        for r in range(math.ceil(len(msbins)/cpus)):
            # print('r', r)
            output_ids = []
            for i in msbins[r*cpus:(r+1)*cpus]:
                # ver3 변경사항
                raw_data2_nmr = raw_data2_storage[:, i:i+binlen]
                raw_data3 = np.array(matlab_FIRfilter(msdata = np.array(raw_data2_nmr), SR=SR)) # FIRfilter 0.5 ~ 50 Hz
                raw_data3 = np.transpose(raw_data3)
                raw_data4 = np.reshape(np.array(raw_data3), (1, raw_data3.shape[0], raw_data3.shape[1]))
                # raw_data4.shape
                output_ids.append(ms_wavelet.remote(xdata=raw_data4, SR=SR, finum=finum))
            output_list = ray.get(output_ids)
            result_concat += output_list
        end = time.time()
    
        fi = finum
        ti = raw_data2_storage.shape[1]
        ch = raw_data2_storage.shape[0]
        
        template = np.zeros((ch, fi, ti)) * np.nan
        template_phase = np.zeros((ch, fi, ti)) * np.nan
        # template_filtered_eeg = np.zeros((ti, fi, ch)) * np.nan
        
        for i in range(len(result_concat)):
            origin = np.array(template[:, :, msbins[i]:msbins[i]+binlen])
            new = np.array(result_concat[i][:,:,:,0,0])
            template[:, :, msbins[i]:msbins[i]+binlen] = np.nanmean([origin, new], axis=0)
            
            origin = np.array(template_phase[:, :, msbins[i]:msbins[i]+binlen])
            new = np.array(result_concat[i][:,:,:,0,1])
            template_phase[:, :, msbins[i]:msbins[i]+binlen] = np.nanmean([origin, new], axis=0)
        return template, template_phase
    
    import ray
    cpus = 6
    ray.shutdown()
    ray.init(num_cpus=cpus)

    template, template_phase \
        = ms_morlet_wavelet(finum=finum, raw_data=EEGdata_ch_x_time, SR=SR, cpus=6)

    return template, template_phase
