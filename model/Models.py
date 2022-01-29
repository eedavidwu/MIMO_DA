import os
from torch.functional import split
import torch.nn.functional as F
import  torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import random


class Fading_Channel(nn.Module):
    def __init__(self, args):
        super(Fading_Channel, self).__init__()
        self.gama=args.gama

    def complex_convolution(self,inputs,h):
        in_shape=inputs.shape
        sub_channel_num=in_shape[2]
        kernel_real=torch.real(h)
        kernel_img=torch.imag(h)


        weight_real=nn.Parameter(data=kernel_real,requires_grad=False)
        weight_img=nn.Parameter(data=kernel_img,requires_grad=False)

        inputs_real=torch.real(inputs).float()
        inputs_img=torch.imag(inputs).float()

        out_r_r=F.conv1d(inputs_real,weight_real,groups=12,padding=sub_channel_num)[:,:,sub_channel_num-sub_channel_num//2:sub_channel_num+sub_channel_num//2,]
        out_r_im=F.conv1d(inputs_real,weight_img,groups=12,padding=sub_channel_num)[:,:,sub_channel_num-sub_channel_num//2:sub_channel_num+sub_channel_num//2,]
        out_im_r=F.conv1d(inputs_img,weight_real,groups=12,padding=sub_channel_num)[:,:,sub_channel_num-sub_channel_num//2:sub_channel_num+sub_channel_num//2,]
        out_im_im=F.conv1d(inputs_img,weight_img,groups=12,padding=sub_channel_num)[:,:,sub_channel_num-sub_channel_num//2:sub_channel_num+sub_channel_num//2,]
        out_real=out_r_r-out_im_im
        out_im=out_r_im+out_im_r
        out=torch.complex(out_real,out_im)
        return out,h
    def compensation(self,inputs,h_broadcast):
        h_norm_square_broadcast=torch.square(torch.abs(h_broadcast))
        h_conj_broadcast=torch.conj(h_broadcast)
        h_norm_square_inverse=torch.where(h_norm_square_broadcast>0,1/h_norm_square_broadcast,h_norm_square_broadcast)

        compensate_matrix=torch.transpose(h_norm_square_inverse*h_conj_broadcast,1,2)
        out=torch.bmm(compensate_matrix,inputs)
        #out=(h_conj_broadcast*inputs)/h_norm_broadcast
        return out
        
    def forward(self, inputs,h,input_snr):
        in_shape=inputs.shape
        batch_size=in_shape[0]

        ##Through fading channel
        complex_para_out=torch.bmm(h,inputs)

        ##awgn:
        noise_stddev=(np.sqrt(10**(-input_snr/10))/np.sqrt(2)).reshape(-1,1,1)
        noise_stddev_board=torch.from_numpy(noise_stddev).repeat(1,in_shape[1],in_shape[2]).cuda()
        mean=torch.zeros_like(noise_stddev_board).cuda()
        noise_real=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda()).float()
        noise_img=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda()).float()
        noise_complex=torch.complex(noise_real,noise_img)

        #idea H:
        H_idea=torch.ones_like(h)
        
        ##y=hx+w
        channel_out=complex_para_out+noise_complex

        ##compensation:
        channel_com_out=self.compensation(channel_out,h)

        new_noise=self.compensation(noise_complex,h)
        new_out=torch.bmm(H_idea,inputs)+new_noise

        return channel_com_out#,inputs_in_norm


class OFDM_trans(nn.Module):
    def __init__(self, args):
        super(OFDM_trans, self).__init__()
        self.CP_num=args.cp_num 
        self.Np=4
        self.Ns=8
        self.clip_flag=args.clip_flag

    def Clip_by_norm_np(self,inputs, peak=8.0):
        '''
        Clip complex number by PAPR
        :param inputs: complex matrix
        :param peak:
        :return:
        '''        
        sig_pwr=torch.square(torch.abs(inputs))+ 1.0e-8
        avg_pwr=sig_pwr.mean()

        clip_condition=peak*avg_pwr

        clip_val = math.sqrt(peak) * inputs / torch.sqrt(sig_pwr)
        outputs = torch.where(sig_pwr < clip_condition, inputs, clip_val)
        return outputs

    def forward(self, Y,Y_p):
        Y_in=torch.cat((Y,Y_p),dim=1)
        #IFFT
        Y_in_IFFT=torch.fft.ifft2(Y_in)
        #add CP:
        ofdm_CP = Y_in_IFFT[:,:,-self.CP_num:]
        ofdm_symbol_cpx = torch.cat((ofdm_CP,Y_in_IFFT),axis=2)
        if self.clip_flag==True:
            y=self.Clip_by_norm_np(ofdm_symbol_cpx)
        else:
            y=ofdm_symbol_cpx
        return y


class OFDM_receiver(nn.Module):
    def __init__(self, args):
        super(OFDM_receiver, self).__init__()
        self.CP_num=args.cp_num 
        self.Np=4
        self.Ns=8
    

    def forward(self, y_head):
        y_head_remove_CP=y_head[:,:,self.CP_num:]
        #FFT
        Y_FFT=torch.fft.fft2(y_head_remove_CP)
        #Y_head=torch.real(Y_FFT[:,0:self.Ns,:]).float()
        #Y_p_head=torch.real(Y_FFT[:,-self.Np:,:]).float()
        Y_head=Y_FFT[:,0:self.Ns,:]
        Y_p_head=Y_FFT[:,-self.Np:,:]

        return Y_head,Y_p_head



class FL_En_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,padding,activation=None):
        super(FL_En_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding)
        self.GDN = nn.BatchNorm2d(out_channels)
        if activation=='sigmoid':
            self.activate_func=nn.Sigmoid()
        elif activation=='prelu':
            self.activate_func=nn.PReLU()
        elif activation==None:
            self.activate_func=None            

    def forward(self, inputs):
        out_conv1=self.conv1(inputs)
        out_bn=self.GDN(out_conv1)
        if self.activate_func != None:
            out=self.activate_func(out_bn)
        else:
            out=out_bn
        return out

class AL_Module(nn.Module):
    def __init__(self,fc_in):
        super(AL_Module, self).__init__()
        self.Ave_Pooling = nn.AdaptiveAvgPool2d(1)
        self.FC_1 = nn.Linear(fc_in+6,fc_in//16)
        self.FC_2 = nn.Linear(fc_in//16,fc_in)

    def forward(self, inputs,h,snr):
        out_pooling=self.Ave_Pooling(inputs).squeeze()
        b=inputs.shape[0]
        c=inputs.shape[1]
        in_fc=torch.cat((snr,h,out_pooling),dim=1).float()
        out_fc_1=self.FC_1(in_fc)
        out_fc_1_relu=torch.nn.functional.relu(out_fc_1)
        out_fc_2=self.FC_2(out_fc_1_relu)
        out_fc_2_sig=torch.sigmoid(out_fc_2).view(b,c,1,1)
        out=out_fc_2_sig*inputs
        return out

class FL_De_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,padding,out_padding,activation=None):
        super(FL_De_Module, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding,output_padding=out_padding)
        self.GDN = nn.BatchNorm2d(out_channels)
        if activation=='sigmoid':
            self.activate_func=nn.Sigmoid()
        elif activation=='prelu':
            self.activate_func=nn.PReLU()
        elif activation==None:
            self.activate_func=None            

    def forward(self, inputs):
        out_deconv1=self.deconv1(inputs)
        out_bn=self.GDN(out_deconv1)
        if self.activate_func != None:
            out=self.activate_func(out_bn)
        else:
            out=out_bn
        return out

class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.FL_Module_1 = FL_En_Module(3, 256, 9, 2, 4,'prelu')
        self.FL_Module_2 = FL_En_Module(256, 256, 5, 2,2, 'prelu')
        self.FL_Module_3 = FL_En_Module(256, 256, 5, 1,2, 'prelu')
        self.FL_Module_4 = FL_En_Module(256, 256, 5, 1,2, 'prelu')
        self.FL_Module_5 = FL_En_Module(256, args.tcn, 5, stride=1,padding=2)

    def forward(self, x):
        encoded_1_out = self.FL_Module_1(x)
        encoded_2_out = self.FL_Module_2(encoded_1_out)
        encoded_3_out = self.FL_Module_3(encoded_2_out)
        encoded_4_out = self.FL_Module_4(encoded_3_out)
        encoded_5_out = self.FL_Module_5(encoded_4_out)
        return encoded_5_out

class Decoder(nn.Module):
    def __init__(self,args):
        super(Decoder, self).__init__()
        self.FL_De_Module_1 = FL_De_Module(16, 256, 5, stride=1,padding=2,out_padding=0,activation='prelu')
        self.FL_De_Module_2 = FL_De_Module(256, 256, 5, stride=1,padding=2,out_padding=0,activation='prelu')
        self.FL_De_Module_3 = FL_De_Module(256, 256, 5, stride=1,padding=2,out_padding=0,activation='prelu')
        self.FL_De_Module_4 = FL_De_Module(256, 256, 5, stride=2,padding=2,out_padding=1,activation='prelu')
        self.FL_De_Module_5 = FL_De_Module(256,3, 9, stride=2,padding=4,out_padding=1,activation='sigmoid')


    def forward(self, x):
        #make the input for decoder:
        #x=torch.cat((Y_p,Y_p_head,Y_head),dim=1).view(encoded_shape[0],-1,encoded_shape[2],encoded_shape[3])

        decoded_1_out = self.FL_De_Module_1(x)
        decoded_2_out = self.FL_De_Module_2(decoded_1_out)
        decoded_3_out = self.FL_De_Module_3(decoded_2_out)
        decoded_4_out = self.FL_De_Module_4(decoded_3_out)
        decoded_5_out = self.FL_De_Module_5(decoded_4_out)
        return decoded_5_out


class Attention_Encoder(nn.Module):
    def __init__(self,args):
        super(Attention_Encoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.FL_Module_1 = FL_En_Module(3, 256, 9, 2, 4,'prelu')
        self.AL_Module_1=AL_Module(256)
        self.FL_Module_2 = FL_En_Module(256, 256, 5, 2,2, 'prelu')
        self.AL_Module_2=AL_Module(256)
        self.FL_Module_3 = FL_En_Module(256, 256, 5, 1,2, 'prelu')
        self.AL_Module_3=AL_Module(256)
        self.FL_Module_4 = FL_En_Module(256, 256, 5, 1,2, 'prelu')
        self.AL_Module_4=AL_Module(256)
        self.FL_Module_5 = FL_En_Module(256, args.tcn, 5, stride=1,padding=2)


    def forward(self, x,h,snr):
        encoded_1_out = self.FL_Module_1(x)
        attention_encoder_1_out=self.AL_Module_1(encoded_1_out,h,snr)

        encoded_2_out = self.FL_Module_2(attention_encoder_1_out)
        attention_encoder_2_out=self.AL_Module_2(encoded_2_out,h,snr)

        encoded_3_out = self.FL_Module_3(attention_encoder_2_out)
        attention_encoder_3_out=self.AL_Module_3(encoded_3_out,h,snr)

        encoded_4_out = self.FL_Module_4(attention_encoder_3_out)
        attention_encoder_4_out=self.AL_Module_4(encoded_4_out,h,snr)

        encoded_5_out = self.FL_Module_5(attention_encoder_4_out)
        return encoded_5_out

class Attention_Decoder(nn.Module):
    def __init__(self,args):
        super(Attention_Decoder, self).__init__()
        self.FL_De_Module_1 = FL_De_Module(16, 256, 5, stride=1,padding=2,out_padding=0,activation='prelu')
        self.AL_De_module_1=AL_Module(256)

        self.FL_De_Module_2 = FL_De_Module(256, 256, 5, stride=1,padding=2,out_padding=0,activation='prelu')
        self.AL_De_module_2=AL_Module(256)

        self.FL_De_Module_3 = FL_De_Module(256, 256, 5, stride=1,padding=2,out_padding=0,activation='prelu')
        self.AL_De_module_3=AL_Module(256)

        self.FL_De_Module_4 = FL_De_Module(256, 256, 5, stride=2,padding=2,out_padding=1,activation='prelu')
        self.AL_De_module_4=AL_Module(256)

        self.FL_De_Module_5 = FL_De_Module(256,3, 9, stride=2,padding=4,out_padding=1,activation='sigmoid')



    def forward(self, x,h,snr):
        decoded_1_out = self.FL_De_Module_1(x)
        attention_decoder_1_out=self.AL_De_module_1(decoded_1_out,h,snr)

        decoded_2_out = self.FL_De_Module_2(attention_decoder_1_out)
        attention_decoder_2_out=self.AL_De_module_2(decoded_2_out,h,snr)

        decoded_3_out = self.FL_De_Module_3(attention_decoder_2_out)
        attention_decoder_3_out=self.AL_De_module_3(decoded_3_out,h,snr)
        
        decoded_4_out = self.FL_De_Module_4(attention_decoder_3_out)
        attention_decoder_4_out=self.AL_De_module_4(decoded_4_out,h,snr)

        decoded_5_out = self.FL_De_Module_5(attention_decoder_4_out)

        return decoded_5_out


class Classic_JSCC(nn.Module):
    def __init__(self,args):
        super(Classic_JSCC, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = Encoder(args)
        #self.transmitter=OFDM_trans(args)
        self.channel=Fading_Channel(args)
        #self.receicer=OFDM_receiver(args)
        self.decoder=Decoder(args)
        self.num_se=2
        self.num_re=2
        self.subchannel_num=80

        #self.gama=args.gama
        #self.compensate=Compensate(args)

    def multipath_rayleigh_fading_stddev(self,num_subchannel, gama):
        result_list=[]
        for i in range (num_subchannel):
            result_list.append(-i/gama)
        result=np.exp(result_list)
        alfa=1/result.sum()
        power_list=result*alfa
        std_list=np.sqrt(power_list)
        #print(power_list)
        #print("sum:",power_list.sum())
        return std_list

    def compute_h_broadcast(self,batch_size,subchannel_num):
        h_stddev=torch.full((batch_size,subchannel_num),1/np.sqrt(2)).float()
        h_stddev = torch.diag_embed(h_stddev)
        h_mean=torch.zeros_like(h_stddev).float()

        h_real=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
        h_img=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
        #h_real=torch.ones_like(h_real).float()
        #h_img=torch.ones_like(h_real).float()
        h=torch.complex(h_real,h_img).cuda()
        return h

    def power_normalize(self,feature):        
        in_shape=feature.shape
        batch_size=in_shape[0]
        sig_pwr=torch.square(torch.abs(feature))
        ave_sig_pwr=sig_pwr.mean(dim=2).unsqueeze(dim=2)
        z_in_norm=feature/(torch.sqrt(ave_sig_pwr))
        return z_in_norm


    def bmm_float_complex(self,float,complex):
        real=torch.real(complex)
        img=torch.imag(complex)
        real_out=torch.bmm(float,real)
        img_out=torch.bmm(float,img)
        return torch.complex(real_out,img_out)


    def transmit_feature(self,feature,channel_snr):
        in_shape=feature.shape
        batch_size=in_shape[0]
        sender_antena=in_shape[1]
        receiver_antena=len(channel_snr)
        each_antena_symbol_num=in_shape[2]
        zero_ave=torch.full((batch_size,receiver_antena,each_antena_symbol_num),0).float()
        feature_ave=torch.complex(zero_ave,zero_ave).cuda()

        ####Power constraint for each sending:
        normalized_X=self.power_normalize(feature)
        #normalized_X=feature
        ##construct input X:
        X=normalized_X

        for i in range (5):
            h_stddev=torch.full((batch_size,receiver_antena,sender_antena),1/np.sqrt(2)).float()
            h_mean=torch.zeros_like(h_stddev).float()
            h_real=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
            h_img=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
            
            #h_real=torch.eye(2,2).unsqueeze(dim=0).repeat(batch_size,1,1).float()
            #h_img=torch.zeros_like(h_real).float()
            H=torch.complex(h_real,h_img).cuda()

            ##SVD of H matrix:
            U, S, Vh = torch.linalg.svd(H, full_matrices=True)
            S_diag=torch.diag_embed(torch.complex(S,torch.zeros_like(S)))
            V = Vh.transpose(-2, -1).conj()
                #U, S, Vh = torch.linalg.svd(H, full_matrices=True)
                #S_diag=torch.diag_embed(torch.complex(S,torch.zeros_like(S)))
                #test_compute=torch.bmm(torch.bmm(U, S_diag), Vh)
                #torch.dist(test_compute,H)

            ##Package input X:
            X_package=torch.bmm(torch.inverse(Vh),X)
            ##Go through channel:
            channel_out_H=torch.bmm(H,X_package)            
            ##awgn: set for same receiver, SNR is same through different MIMO channels
            one_receiver_antena_AWGN_noise_shape=X.shape

            for k in range(len(channel_snr)):
                snr=channel_snr[k]
                noise_stddev=(np.sqrt(10**(-snr/10))/np.sqrt(2)).reshape(1,1,1)
                #noise_stddev=np.zeros((1,1,1))
                noise_stddev_board=torch.from_numpy(noise_stddev).repeat(one_receiver_antena_AWGN_noise_shape[0],1,one_receiver_antena_AWGN_noise_shape[2]).float().cuda()
                mean=torch.zeros_like(noise_stddev_board).float().cuda()

                w_real=Variable(torch.normal(mean=mean,std=noise_stddev_board)).float()
                w_img=Variable(torch.normal(mean=mean,std=noise_stddev_board)).float()

                noise_for_one_receiver=torch.complex(w_real,w_img)
                
                if k==0:
                    W=noise_for_one_receiver
                else:
                    W=torch.cat((W,noise_for_one_receiver),dim=1)
                
            channel_out_H_W=channel_out_H+W
            ##Compensation:
            Y=(torch.bmm(torch.inverse(U),channel_out_H_W))
            #Y_o=torch.bmm(S_diag,X)+(torch.bmm(torch.inverse(U),W))
            feature_ave=feature_ave+Y
        feature_out=feature_ave/5
        return feature_out
    


    def forward(self, x,input_snr):
        #prepare parameter:
        batch_size=x.shape[0]
        N_s=2
        N_r=2
        ##To do more times to avoid randomness:
        #papr_ave=torch.zeros(batch_size,12).float().cuda()
        #encode
        encoded_out = self.encoder(x)
        encoded_shape=encoded_out.shape

        #reshape
        batch_size=encoded_shape[0]
        z=encoded_out.view(batch_size,-1)
        complex_list=torch.split(z,(z.shape[1]//2),dim=1)
        encoded_out_complex=torch.complex(complex_list[0],complex_list[1])
        Y=encoded_out_complex.view(batch_size,N_s,-1)
        #compute PAPR
        #trans_y_power=torch.square(torch.abs(trans_y))
        #ave_trans_y_power=torch.mean(trans_y_power,dim=2)
        #papr=torch.max(trans_y_power,dim=2)[0]/ave_trans_y_power
        #h_attention=torch.cat((h_real_attention,h_img_attention))
        y_head_=self.transmit_feature(Y,input_snr)
        #y_head_=Y

        #make the input for decoder:
        Y_all_antena=y_head_.view(batch_size,-1)
        Y_head_de=torch.cat((torch.real(Y_all_antena),torch.imag(Y_all_antena)),dim=1)
        decoder_input=Y_head_de.view(batch_size,-1,encoded_shape[2],encoded_shape[3]).float()
        
        
        #Channel estimation
        x_head= self.decoder(decoder_input)

        #papr_ave=papr+papr_ave
        #x_head_ave=x_head+x_head_ave
        #papr_ave=papr_ave/5
        #x_head_ave=x_head_ave/5
        return x_head

class Attention_all_JSCC(nn.Module):
    def __init__(self,args):
        super(Attention_all_JSCC, self).__init__()
        self.attention_encoder = Attention_Encoder(args)
        self.channel=Fading_Channel(args)
        self.attention_decoder=Attention_Decoder(args)
        self.gama=args.gama
        self.transmitter=OFDM_trans(args)
        self.receicer=OFDM_receiver(args)
        self.SNR_min=args.input_snr_min
        self.SNR_max=args.input_snr_max
        self.h_phase_error_max=args.h_error_max
        self.h_phase_error_min=args.h_error_min

        self.num_se=2
        self.num_re=2
        self.estimation_error=args.h_error_flag
        self.decompose_flag=args.h_decom_flag

    
    def power_normalize(self,feature):        
        sig_pwr=torch.square(torch.abs(feature))
        ave_sig_pwr=sig_pwr.mean(dim=2).unsqueeze(dim=2)
        z_in_norm=feature/(torch.sqrt(ave_sig_pwr))
        return z_in_norm

        return inputs_in_norm
    def bmm_float_complex(self,float,complex):
        real=torch.real(complex)
        img=torch.imag(complex)
        real_out=torch.bmm(float,real)
        img_out=torch.bmm(float,img)
        return torch.complex(real_out,img_out)

    def forward(self,x,input_snr):
        #prepare parameter:
        batch_size=x.shape[0]
        N_s=2
        N_r=2
        #reshape
        x_head_ave=torch.zeros_like(x).cuda()

        for transmit_times in range (5):
            if input_snr=='random_in_list':
                snr_1=np.random.rand(batch_size,)*(self.SNR_max-self.SNR_min)+self.SNR_min
                snr_2=np.random.rand(batch_size,)*(self.SNR_max-self.SNR_min)+self.SNR_min
            else:
                snr_1=np.full(batch_size,input_snr[0])
                snr_2=np.full(batch_size,input_snr[1])

            channel_snr=np.stack((snr_1,snr_2),axis=1)
            channel_snr_attention=torch.from_numpy(channel_snr).float().cuda()
            #prepare H at each time for the attention and channel:
            h_stddev=torch.full((batch_size,N_r,N_s),1/np.sqrt(2)).float()
            h_mean=torch.zeros_like(h_stddev).float()
            h_real=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
            h_img=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
            H=torch.complex(h_real,h_img).cuda()
            #H:(-1,2,2)
            if self.decompose_flag==1:
            ##SVD of H matrix:
                U, S, Vh = torch.linalg.svd(H, full_matrices=True)
                S_diag=torch.diag_embed(torch.complex(S,torch.zeros_like(S)))
                #if perfect:
                Vh_est=Vh
                S_est=S
                U_est=U
                S_diag_est=S_diag

                #if estimate wrong:
                if (self.estimation_error)!=0:
                    phase_error=np.random.rand(batch_size,N_r,N_s)*(self.h_phase_error_max-self.h_phase_error_min)+self.h_phase_error_min
                    h_error=torch.complex(torch.from_numpy(np.cos(phase_error)).float(),torch.from_numpy(np.sin(phase_error)).float()).cuda()
                    H_est=h_error*H
                    U_est, S_est, Vh_est = torch.linalg.svd(H_est, full_matrices=True)
                    S_diag_est=torch.diag_embed(torch.complex(S_est,torch.zeros_like(S)))

                h_est_attention=S_est
                h_real_attention=S
            else:
                h_real_attention=H.view(batch_size,-1)
                h_est_attention=h_real_attention
                
            #real Channel Use H
            #attention use H_est

            #encode
            encoded_out = self.attention_encoder(x,h_est_attention,channel_snr_attention)
            encoded_shape=encoded_out.shape
            batch_size=encoded_shape[0]
            z=encoded_out.view(batch_size,-1)
            complex_list=torch.split(z,(z.shape[1]//2),dim=1)
            encoded_out_complex=torch.complex(complex_list[0],complex_list[1])
            Y=encoded_out_complex.view(batch_size,N_s,-1)
            num_sym=Y.shape[2]
            ####Power constraint for each sending:
            normalized_X=self.power_normalize(Y)
            ##constrcut AWGN:
            for k in range(self.num_re):
                snr=channel_snr[:,k]
                noise_stddev=(np.sqrt(10**(-snr/10))/np.sqrt(2)).reshape(-1,1,1)
                noise_stddev_board=torch.from_numpy(noise_stddev).repeat(1,1,num_sym).float()
                mean=torch.zeros_like(noise_stddev_board).float()
                
                noise_real=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda()).float()
                noise_img=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda()).float()
                noise_complex_for_one_sender=torch.complex(noise_real,noise_img)
                if k==0:
                    W=noise_complex_for_one_sender
                else:
                    W=torch.cat((W,noise_complex_for_one_sender),dim=1)
            #transmit:
            '''
            #transmit_style_1:
            X_package=torch.bmm(torch.inverse(Vh),normalized_X)
            channel_out_H=torch.bmm(H,X_package)   
            channel_out_H_W=channel_out_H+W
            Y=(torch.bmm(torch.inverse(U),channel_out_H_W))
            '''
            if self.decompose_flag==1:
            #transmit_style_2:
                Y=torch.bmm(S_diag,normalized_X)+torch.bmm(torch.inverse(U),W)
            else:
                Y=torch.bmm(H,normalized_X)+W
                
            ##Directly:
            #Y=torch.bmm(S_diag,X)+torch.bmm(torch.inverse(U),W)
            #Decoder: make the input for decoder:
            Y_all_antena=Y.view(batch_size,-1)
            Y_head_de=torch.cat((torch.real(Y_all_antena),torch.imag(Y_all_antena)),dim=1)
            decoder_input=Y_head_de.view(batch_size,-1,encoded_shape[2],encoded_shape[3]).float()
            if (self.estimation_error)==1:
                ##1 all wrong esti
                x_head= self.attention_decoder(decoder_input,h_est_attention,channel_snr_attention)
            else:
                ##1 receiver right
                x_head= self.attention_decoder(decoder_input,h_real_attention,channel_snr_attention)
            #Transmit 5 times:    
            x_head_ave=x_head+x_head_ave
        x_head_ave=x_head_ave/5

        return x_head_ave
