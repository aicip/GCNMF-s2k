
%% Load Disaggregation Demo code. 
%Alireza Rahimpour. Contact: arahimpo@utk.edu
%  
%
%
% This code use different algorithms for energy disaggregation task.
% ***different modes: 
% 
% ****Training mode: parameter: training=1 
% In this mode there is no prediction of energy and it's just signal decomposition
% we can choose days from 1 to k in this mode. 
% 
% 
% ****Testing mode: parameter: testing=1
% k is the number of days used for each device in the dictionary. 
% number of days for testing is: en-st  en: ending day. st: staring day.
% 
% 
% ****Using Week instead of days as one base in the dictionary. 
% parameter: week=1
% 
% *****Choosing the data: Between real aggregated signal and synthetic( sum of x_i s)
% parameter: real_agg=1
% 
% 
% *****Normalization: using normalized version of all the signals. 
% parameter: Normalization=1
% 
% 
% ******using Non negative sparse coding algorithm ( not using SPAMS) 
% parameter: spc=1
% 
% ******Using SPAMS for implementing the 'Elastic net' and L2 norm algorithms.
% parameters:
% SPAMS=0;
% % Choose the algorithm: ( just one of these can be 1) 
% grouplasso=0; ( not done yet)
% Elastic_net=0;
% L2norm=1-Elastic_net;
% 
% *****USING NNLS and SUM to one constraint for each device:
% ***** fast=1 USE THE FAST NNLS !
% *****NNLS_code=1 
% For using the sum to one constraint: constraint1=1
% 
% *****Average of current of each device:
% Minimization of average of current of each device and its estimate. Parameter: constraint2=1
% 
% *****Post processing: Limiting the estimated signal to be inside the max and min of the signal ( based on hostory of usage) 
% However, this post processing doesn't have significant effect on result. 
% 
% **********OUTPUT: 
% RMSE for each device and aggregated signal. Plot of the graound truth and estimated signals. 
% Sum of energy( current) for each day for true and estimated signal. 
% 
% if show_all=1 plot all the devices in 2 figures. ( estimated and true signal) 
%p= number of devices 

%if  plot_agg==1: Plots the aggregated signal and its estimate.

%##### Written by: Alireza Rahimpour. arahimpo@utk.edu#####
% Best result achieve by using : NNLS algorithm and sum to one constraint
% for each device. ( using k=7 and next day as testing day) 
% Best aggregated signal estimation: without sum to one constraint and with
% large k. 

%%
%DATA 
clear all; 
close all;
load('AMP_DATA.mat') % this dataset has 364 days of data and each column is one day. 
real_agg=1; % real aggregated signal or synthetic one
X=x_agg;
 

%%
%TRAINING 
% In training just use k from 1 to K. ( because of structure of A)
training=1;
testing=1-training;
plot_agg=0;
if training==1
    display('****BASIC Training****');
     k=1; % value of k for training
     display( strcat(['number of training days:',num2str(k)]));
     constraint=0;
    %st is k+1 ( start) and en is end. 
st=1; % between 1 and K 
en=k;  % these two numbers should be the same for A and X. 
 constraint1=1;% for sum to one constraint
     constraint2=0; % for average constraint
else
    %%
    %Adjusting the parameters for TESTING:
    
    k=7; % using k days for training and 364-k days for testing. In case of week==1: max k is 52. k=7 ( one week ) 
    testing=1;
    display('****TESTING****');
    st=k+1;
    en=st; %end
     constraint1=1;% for sum to one constraint
     constraint2=0; % for average constraint
end
%%
%changing each column to one week instead of one day: 
week=0;
if (week|training)==0
display( strcat(['number of training days:',num2str(k)]));
display( strcat(['number of testing days:',num2str(en-k)]));
end
if week==1  %Now each column is ONE WEEK
    display('Using one WEEK for each column');
    display( strcat(['number of training WEEKS:',num2str(k)]));
    display( strcat(['number of testing WEEKS:',num2str(52-k)]));
X1=reshape(X1,1440*7,364/7);
x1=reshape(x1,1440*7,364/7); x2=reshape(x2,1440*7,364/7);x3=reshape(x3,1440*7,364/7);x4=reshape(x4,1440*7,364/7);x5=reshape(x5,1440*7,364/7);x6=reshape(x6,1440*7,364/7);
x7=reshape(x7,1440*7,364/7);x8=reshape(x8,1440*7,364/7);x9=reshape(x9,1440*7,364/7);x10=reshape(x10,1440*7,364/7);x11=reshape(x11,1440*7,364/7);x12=reshape(x12,1440*7,364/7);
x13=reshape(x13,1440*7,364/7);x14=reshape(x14,1440*7,364/7);x15=reshape(x15,1440*7,364/7);x16=reshape(x16,1440*7,364/7);x17=reshape(x17,1440*7,364/7);x18=reshape(x18,1440*7,364/7);
x19=reshape(x19,1440*7,364/7);
end

%%
%normalization
  normalization=0;

if normalization==1
    %%normalization of basis by norm of aggregated signal
    display('NORMALIZATION: ON');
    x1=x1/norm(X);
    x2=x2/norm(X);
    x3=x3/norm(X);
    x4=x4/norm(X);
    x5=x5/norm(X);
    x6=x6/norm(X);
    x7=x7/norm(X);
    x8=x8/norm(X);
    
    x9=x9/norm(X);
    x10=x10/norm(X);
    x11=x11/norm(X);
    x12=x12/norm(X);
    x13=x13/norm(X);
    x14=x14/norm(X);
    x15=x15/norm(X);
    x16=x16/norm(X);
    x17=x17/norm(X);
    x18=x18/norm(X); 
    x19=x19/norm(X);
    
    X =X/norm(X);
else  display('NORMALIZATION: OFF');
end
%%
%USING real physical signal or synthetic data 
if real_agg==1
    X=X(:,st:en);  %%% using real AGG signal
    display('USING REAL AGG SIGNAL');
else
     X=x1(:,st:en)+x2(:,st:en)+x3(:,st:en)+x4(:,st:en)+x5(:,st:en)+x6(:,st:en)+x7(:,st:en)+x8(:,st:en)+x9(:,st:en)+x10(:,st:en)+x11(:,st:en)+x12(:,st:en)+x13(:,st:en)+x14(:,st:en)+x15(:,st:en)+x16(:,st:en)+x17(:,st:en)+x18(:,st:en)+x19(:,st:en);
    display('USING SUM OF SIGNLAS AS AGG SIGNAL');
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                   TESTING
%%Concatinating A
A_concat=[x1(:,1:k),x2(:,1:k),x3(:,1:k),x4(:,1:k),x5(:,1:k),x6(:,1:k),x7(:,1:k),x8(:,1:k),x9(:,1:k),x10(:,1:k),x11(:,1:k),x12(:,1:k),x13(:,1:k),x14(:,1:k),x15(:,1:k),x16(:,1:k),x17(:,1:k),x18(:,1:k),x19(:,1:k)];  % I think dimension A is arbitary
A=A_concat;
%%
showflag = 1;
mixed=X;
 [M,N]=size(X);
%% Test different algorithms
%% sparse coding 
spc=0;% if you want to use sparsecoding algorithm then: spc=1
NNLS_code=1-spc;%%%use this command for using FCLS code1

%%
SPAMS=0;
% Choose the algorithm: ( just one of these can be 1) 
grouplasso=0;
Elastic_net=1;
L2norm=1-Elastic_net;

if SPAMS==1

    NNLS_code=0;
    spc=0;
    display('USING SPAMS') 
    
    start_spams
    get_architecture
    
     %%
    %sum to one cons ( Not compatible with elastic net and L2 norm ) 
    % Adding sum to one constraint for each device to group lasso algorithm
    sum_to_one_spams=0;
    if sum_to_one_spams==1
    p=19;
        for i=1:(p-1)       
            c(i,:)=[zeros(1,i*k),ones(1,k),zeros(1,(p-(i+1))*k)];
        end
    c=[ones(1,k),zeros(1,(p-1)*k);c];
    AA = [1e-5*A;c]; % ! We can adjust the coefficient of A 
    XX = [1e-5*X;ones(p,1)];
    end
    
%set the parameters:
%     param.numThreads=-1; % all cores (-1 by default)
%     param.verbose=true; % verbosity, false by default
%     param.lambda=0.05; % regularization parameter
%     param.it0=10; % frequency for duality gap computations
%     param.max_it=200; % maximum number of iterations
%     param.L0=0.1;
%     param.tol=1e-3;
%     param.intercept=false;
   
%     param.loss='square';
%     W0=zeros(size(A,2),size(X,2));


    % parameter of the optimization procedure are chosen
    %param.L=20; % not more than 20 non-zeros coefficients (default: min(size(D,1),size(D,2)))
   param.pos=true; % positivity constraint
   param.numThreads=-1; % number of processors/cores to use; the
    
    %group lasso
    
 if grouplasso==1  % No efficient ALG is available yet. 
    
    fprintf('\nGroup Lasso ADMM ALGORITHM\n');
   
   

 else if  Elastic_net==1  
         

        display('ALG: ELASTIC NET');
        param.lambda=0.001; % not more than 20 non-zeros( for lambada=0.15) coefficients
        param.lambda2=10; %(optional parameter for solving the Elastic-Net) 
         % for mode=0 and mode=1, it adds a ridge on the Gram Matrix
        sest=mexLasso(X,A,param);
         
     else if L2norm==1
               % 3) when param.mode=2
    % min_{alpha} 0.5||x-Dalpha||_2^2 + lambda||alpha||_1 +0.5 lambda2||alpha||_2^2

    % param.lambda2 (optional parameter for solving the Elastic-Net)
    % for mode=0 and mode=1, it adds a ridge on the Gram Matrix
             display('ALG:L2 NORM')
            param.lambda=0; % L1 not more than 20 non-zeros( for lambada=0.15) coefficients
            param.lambda2=0.001; %L2 (optional parameter for solving the Elastic-Net) 
            % for mode=0 and mode=1, it adds a ridge on the Gram Matrix
            sest=mexLasso(X,A,param);
             
         end
     end
 end 
end
%%
    

if spc==1
   
    display('Algorithm: SPARSE CODING (not using SPAMS BTW)')
    display( 'Processing... It may take a few minutes... :) ')

   %%% this is how we should define variables for sparsecoding algorithm.
   %%% b=Ax which : b=A'*X  and  AA=A'*A;  A: your original dictionary and
   %%% X is your original observation.
   
    b=A'*X;
    AA=A'*A;
    
    lambada=0.0001;
    
        for j=1:N  %mn
        r=b(:,j);
        s_sc(:,j) = sparsecoding( lambada, AA, r );
        sest = s_sc;
        end
       
 end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FCLS code 1
if NNLS_code==1
     display('Algorithm: NLS')
     display( 'Processing... It may take a few minutes... :) ')

     p=19; %num of devices
     
 if (constraint1|constraint2)==0
     display('No constraint');
        warning off;
        
    %fast NNLS
   fast=0;
   if fast==1
      display('FAST NNLS');
%   it solves the problem min ||y - Xx|| if
%	A default tolerance of TOL = MAX(SIZE(XtX)) * NORM(XtX,1) * EPS
%	is used for deciding when elements of x are less than zero.
%	This can be overridden with x = fnnls(XtX,Xty,TOL).
    XtX = A'*A ; 
    Xty =A'*X;
    sest = fnnls(XtX,Xty);
    
   else
       display('Regular NNLS');
        %AA = [1e-5*A_concat;ones(1,length(A_concat(1,:)))];
        AA=A_concat;
        s_fcls = zeros(length(A_concat(1,:)),N);%MN= NUMBER OF COLUMNS
      for j=1:N  %mn
        %r = [1e-5*X(:,j); 1];
        r=X(:,j);
%       s_fcls(:,j) = nnls(AA,r);
        s_fcls(:,j) = lsqnonneg(AA,r);
        sest = s_fcls;
      end
   end
  else if constraint1==1
     display('Sum to 1 constraint for each device( No average constraint)')
     constraint2=0;
%% Sum to one constraint for each device:

    for i=1:(p-1)
           
    c(i,:)=[zeros(1,i*k),ones(1,k),zeros(1,(p-(i+1))*k)];

    end
    c=[ones(1,k),zeros(1,(p-1)*k);c];

    AA = [1e-5*A;c]; % ! We can adjust the coefficient of A 
    % s_fcls = zeros(length(A(1,:)),N);
    
      for j=1:N
    r = [1e-5*X(:,j);ones(p,1)];
    %   s_fcls(:,j) = nnls(AA,r);
    sest(:,j) = lsqnonneg(AA,r);
   
      end
      end 
 end
end
%%
% Average of the signal constraint + sum to one :

    if constraint2==1
            display('Imposing average constraint and sum to one ');
              q=19; %number of devices in which we want to use average constraint for. 

          for i=0:(q-1)
           
            d((i*M)+1:(i+1)*M,:)=[zeros(M,i*k),A(:,(i*k)+1:(i+1)*k),zeros(M,(p-(i+1))*k)];

          end
  

    for j=0:(q-1)
         for i=1:M
           mu(i,:)=mean(A(i,(j*k)+1:(j+1)*k)); 
         end
    H((j*M)+1:(j+1)*M,:)=mu;
    end

%B and mu for augmanting X and A for imposing the mean of signal
%constraint. 

    for i=1:(p-1)
           
         c(i,:)=[zeros(1,i*k),ones(1,k),zeros(1,(p-(i+1))*k)];

    end
         c=[ones(1,k),zeros(1,(p-1)*k);c];

            AA = [1e-5*A;c;d]; % ! We can adjust the coefficient of A 
             % s_fcls = zeros(length(A(1,:)),N);
         for j=1:N
         r = [1e-5*X(:,j);ones(p,1);H];
%       s_fcls(:,j) = nnls(AA,r);
         sest(:,j) = lsqnonneg(AA,r);
   
         end
   end
 
 
%%
display('**************Summary of results***************** ');
display('************************************************* ');
display('         *************************** ');


  n_nonzeroes=nnz(sest);
  display( strcat(['Number of non-zero elements in Sest=',num2str(n_nonzeroes)]))
    %figure, stem(sest);  
   if testing==1 
     if week==1
      E=52-k;
        else
      E=en-k;
      DAY=E; 
     end
  else
      DAY=k;
  end
%   
%or manualy define DAY(should be less than (364-K) :
%DAY=1;
%Also don't forget to change U
            y=A(1:M,:)*sest;
            X_est=y;
            
            %applying max-min on aggregated signal
            max_min=0;
           
     if max_min==1
         display('Applying bound(Max/Min) constraint-post processing') 
        MAX_AGG=max(X(:));
        MIN_AGG=min(X(:));
        %[Q,I] = max(y(:)); Q=the value, I=Index of that value. 
        y(y> MAX_AGG) = MAX_AGG;
        y(y< MIN_AGG) = MIN_AGG;                                    
     end      
            
            
            RMSE_AGG=rmse(X(:,1:DAY),y(:,1:DAY));
            display( strcat(['RMSE_AGG=',num2str(RMSE_AGG)]))
            %y=y';
            %z=Aest*sest;
            
            if plot_agg==1
            figure, 
subplot(3,1,1), plot(X(:,1:DAY),'linewidth',2);title('Aggregated signal (ground truth)'); xlabel('Time(MIN)'), ylabel('Current(AMP)');
        subplot(3,1,2), plot(X_est(:,1:DAY),'linewidth',2);title('Esitimated Aggregated Signal');
        xlabel('Time(MIN)'), ylabel('Current(AMP) ');
                subplot(3,1,3), plot(abs(X(:,1:DAY)-X_est(:,1:DAY)),'linewidth',2);title('Absolute Error');
        xlabel('Time(MIN)'), ylabel('Error ');
            end
s1=sum(y(:,1:DAY));
s2=sum(X(:,1:DAY));
display( strcat(['P_est_total = ',num2str(s1),'  and   p_true_total= ',num2str(s2)]))
display('#')
      
x_est1=x1(:,1:k)*sest(1:k,:);
% a little post processing
MAX_X=max(x1(:)); % it was better to write x1 only on training set
MIN_X=min(x1(:));
x_est1(x_est1>MAX_X)=MAX_X;
x_est1(x_est1<MIN_X)=MIN_X;
%
RMSE1=rmse(x1(:,st:en),x_est1);
display( strcat(['RMSE1=',num2str(RMSE1)]))
s1=sum(x_est1);  % Calculating the sum of the current for each day separately. 
S1=sum(x1(:,st:en)); 
% display( strcat(['P_est = ',num2str(s1),'  and   p_true= ',num2str(s2)]))
p1=(s1/sum(y(:,1:DAY)))*100;
P1=(S1/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 1 is ',num2str(p1),'% of the whole and for ground truth is  = ',num2str(P1),'%']))
display('#')
N1=norm(x1(:,st:en)-x_est1);

x_est2=x2(:,1:k)*sest(k+1:2*k,:);

MAX_X=max(x2(:)); % it was better to write x1 only on training set
MIN_X=min(x2(:));
x_est2(x_est2>MAX_X)=MAX_X;
x_est2(x_est2<MIN_X)=MIN_X;

RMSE2=rmse(x2(:,st:en),x_est2);
display( strcat(['RMSE2=',num2str(RMSE2)]))

s2=sum(x_est2);
S2=sum(x2(:,st:en));
% display( strcat(['P_est2 = ',num2str(s1),'  and   p_true2= ',num2str(s2)]))
p2=(s2/sum(y(:,1:DAY)))*100;
P2=(S2/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 2 is ',num2str(p2),'% of the whole and for ground truth is  = ',num2str(P2),'%']))
display('#')
N2=norm(x2(:,st:en)-x_est2);

x_est3=x3(:,1:k)*sest(2*k+1:3*k,:);

MAX_X=max(x3(:)); 
MIN_X=min(x3(:));
x_est3(x_est3>MAX_X)=MAX_X;
x_est3(x_est3<MIN_X)=MIN_X;

RMSE3=rmse(x3(:,st:en),x_est3);
display( strcat(['RMSE3=',num2str(RMSE3)]))
s3=sum(x_est3);
S3=sum(x3(:,st:en));
% display( strcat(['P_est3 = ',num2str(s1),'  and   p_true3= ',num2str(s2)]))
p3=(s3/sum(y(:,1:DAY)))*100;
P3=(S3/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 3 is ',num2str(p3),'% of the whole and for ground truth is  = ',num2str(P3),'%']))
display('#')
N3=norm(x3(:,st:en)-x_est3);

x_est4=x4(:,1:k)*sest(3*k+1:4*k,:);

MAX_X=max(x4(:));
MIN_X=min(x4(:));
x_est4(x_est4>MAX_X)=MAX_X;
x_est4(x_est4<MIN_X)=MIN_X;

RMSE4=rmse(x4(:,st:en),x_est4);
display( strcat(['RMSE4=',num2str(RMSE4)]))
s4=sum(x_est4);
S4=sum(x4(:,st:en));
% display( strcat(['P_est4 = ',num2str(s1),'  and   p_true4= ',num2str(s2)]))
p4=(s4/sum(y(:,1:DAY)))*100;
P4=(S4/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 4 is ',num2str(p4),'% of the whole and for ground truth is  = ',num2str(P4),'%']))
display('#')
N4=norm(x4(:,st:en)-x_est4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_est5=x5(:,1:k)*sest(4*k+1:5*k,:);

MAX_X=max(x5(:)); 
MIN_X=min(x5(:));
x_est5(x_est5>MAX_X)=MAX_X;
x_est5(x_est5<MIN_X)=MIN_X;

RMSE5=rmse(x5(:,st:en),x_est5);
display( strcat(['RMSE5=',num2str(RMSE5)]))
s5=sum(x_est5);
S5=sum(x5(:,st:en));
% display( strcat(['P_est5 = ',num2str(s1),'  and   p_true5= ',num2str(s2)]))
p5=(s5/sum(y(:,1:DAY)))*100;
P5=(S5/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 5 is ',num2str(p5),'% of the whole and for ground truth is  = ',num2str(P5),'%']))
display('#')
N5=norm(x5(:,st:en)-x_est5);

x_est6=x6(:,1:k)*sest(5*k+1:6*k,:);

MAX_X=max(x6(:)); 
MIN_X=min(x6(:));
x_est6(x_est6>MAX_X)=MAX_X;
x_est6(x_est6<MIN_X)=MIN_X;

RMSE6=rmse(x6(:,st:en),x_est6);
display( strcat(['RMSE6=',num2str(RMSE6)]))
s6=sum(x_est6);
S6=sum(x6(:,st:en));
% display( strcat(['P_est6= ',num2str(s1),'  and   p_true6= ',num2str(s2)]))
p6=(s6/sum(y(:,1:DAY)))*100;
P6=(S6/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 6 is ',num2str(p6),'% of the whole and for ground truth is  = ',num2str(P6),'%']))
display('#')
N6=norm(x6(:,st:en)-x_est6);

x_est7=x7(:,1:k)*sest(6*k+1:7*k,:);

MAX_X=max(x7(:)); 
MIN_X=min(x7(:));
x_est7(x_est7>MAX_X)=MAX_X;
x_est7(x_est7<MIN_X)=MIN_X;

RMSE7=rmse(x7(:,st:en),x_est7);
display( strcat(['RMSE7=',num2str(RMSE7)]))
s7=sum(x_est7);
S7=sum(x7(:,st:en));
% display( strcat(['P_est7 = ',num2str(s1),'  and   p_true7= ',num2str(s2)]))
p7=(s7/sum(y(:,1:DAY)))*100;
P7=(S7/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 7 is ',num2str(p7),'% of the whole and for ground truth is  = ',num2str(P7),'%']))
display('#')
N7=norm(x7(:,st:en)-x_est7);

x_est8=x8(:,1:k)*sest(7*k+1:8*k,:);

MAX_X=max(x8(:)); 
MIN_X=min(x8(:));
x_est8(x_est8>MAX_X)=MAX_X;
x_est8(x_est8<MIN_X)=MIN_X;

RMSE8=rmse(x8(:,st:en),x_est8);
display( strcat(['RMSE8=',num2str(RMSE8)]))
s8=sum(x_est8);
S8=sum(x8(:,st:en));
% display( strcat(['P_est8 = ',num2str(s1),'  and   p_true8= ',num2str(s2)]))
p8=(s8/sum(y(:,1:DAY)))*100;
P8=(S8/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 8 is ',num2str(p8),'% of the whole and for ground truth is  = ',num2str(P8),'%']))
display('#')
N8=norm(x8(:,st:en)-x_est8);

x_est9=x9(:,1:k)*sest(8*k+1:9*k,:);

MAX_X=max(x9(:)); 
MIN_X=min(x9(:));
x_est9(x_est9>MAX_X)=MAX_X;
x_est9(x_est9<MIN_X)=MIN_X;

RMSE9=rmse(x9(:,st:en),x_est9);
display( strcat(['RMSE9=',num2str(RMSE9)]))
s9=sum(x_est9);
S9=sum(x9(:,st:en));
% display( strcat(['P_est9 = ',num2str(s1),'  and   p_true9= ',num2str(s2)]))
p9=(s9/sum(y(:,1:DAY)))*100;
P9=(S9/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 9 is ',num2str(p9),'% of the whole and for ground truth is  = ',num2str(P9),'%']))
display('#')
N9=norm(x9(:,st:en)-x_est9);

x_est10=x10(:,1:k)*sest(9*k+1:10*k,:);

MAX_X=max(x10(:)); 
MIN_X=min(x10(:));
x_est10(x_est10>MAX_X)=MAX_X;
x_est10(x_est10<MIN_X)=MIN_X;

RMSE10=rmse(x10(:,st:en),x_est10);
display( strcat(['RMSE10=',num2str(RMSE10)]))
s10=sum(x_est10);
S10=sum(x10(:,st:en));
% display( strcat(['P_est10 = ',num2str(s1),'  and   p_true= ',num2str(s2)]))
p10=(s10/sum(y(:,1:DAY)))*100;
P10=(S10/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 10 is ',num2str(p10),'% of the whole and for ground truth is  = ',num2str(P10),'%']))
display('#')
N10=norm(x10(:,st:en)-x_est10);

x_est11=x11(:,1:k)*sest(10*k+1:11*k,:);

MAX_X=max(x11(:)); 
MIN_X=min(x11(:));
x_est11(x_est11>MAX_X)=MAX_X;
x_est11(x_est11<MIN_X)=MIN_X;

RMSE11=rmse(x11(:,st:en),x_est11);
display( strcat(['RMSE11=',num2str(RMSE11)]))
s11=sum(x_est11);
S11=sum(x11(:,st:en));
% display( strcat(['P_est 11= ',num2str(s1),'  and   p_true= ',num2str(s2)]))
p11=(s11/sum(y(:,1:DAY)))*100;
P11=(S11/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 11 is ',num2str(p11),'% of the whole and for ground truth is  = ',num2str(P11),'%']))
display('#')
N11=norm(x11(:,st:en)-x_est11);

x_est12=x12(:,1:k)*sest(11*k+1:12*k,:);

MAX_X=max(x12(:)); 
MIN_X=min(x12(:));
x_est12(x_est12>MAX_X)=MAX_X;
x_est12(x_est12<MIN_X)=MIN_X;

RMSE12=rmse(x12(:,st:en),x_est12);
display( strcat(['RMSE12=',num2str(RMSE12)]))
s12=sum(x_est12);
S12=sum(x12(:,st:en));
% display( strcat(['P_est = ',num2str(s1),'  and   p_true= ',num2str(s2)]))
p12=(s12/sum(y(:,1:DAY)))*100;
P12=(S12/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 12 is ',num2str(p12),'% of the whole and for ground truth is  = ',num2str(P12),'%']))
display('#')
N12=norm(x12(:,st:en)-x_est12);

x_est13=x13(:,1:k)*sest(12*k+1:13*k,:);

MAX_X=max(x13(:)); 
MIN_X=min(x13(:));
x_est13(x_est13>MAX_X)=MAX_X;
x_est13(x_est13<MIN_X)=MIN_X;

RMSE13=rmse(x13(:,st:en),x_est13);
display( strcat(['RMSE13=',num2str(RMSE13)]))
s13=sum(x_est13);
S13=sum(x13(:,st:en));
% display( strcat(['P_est 13= ',num2str(s1),'  and   p_true= ',num2str(s2)]))
p13=(s13/sum(y(:,1:DAY)))*100;
P13=(S13/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 13 is ',num2str(p13),'% of the whole and for ground truth is  = ',num2str(P13),'%']))
display('#')
N13=norm(x13(:,st:en)-x_est13);

x_est14=x14(:,1:k)*sest(13*k+1:14*k,:);

MAX_X=max(x14(:)); 
MIN_X=min(x14(:));
x_est14(x_est14>MAX_X)=MAX_X;
x_est14(x_est14<MIN_X)=MIN_X;

RMSE14=rmse(x14(:,st:en),x_est14);
display( strcat(['RMSE14=',num2str(RMSE14)]))
s14=sum(x_est14);
S14=sum(x14(:,st:en));
% display( strcat(['P_est14 = ',num2str(s1),'  and   p_true= ',num2str(s2)]))
p14=(s14/sum(y(:,1:DAY)))*100;
P14=(S14/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 14 is ',num2str(p14),'% of the whole and for ground truth is  = ',num2str(P14),'%']))
display('#')
N14=norm(x14(:,st:en)-x_est14);

x_est15=x15(:,1:k)*sest(14*k+1:15*k,:);

MAX_X=max(x15(:)); 
MIN_X=min(x15(:));
x_est15(x_est15>MAX_X)=MAX_X;
x_est15(x_est15<MIN_X)=MIN_X;

RMSE15=rmse(x15(:,st:en),x_est15);
display( strcat(['RMSE15=',num2str(RMSE15)]))
s15=sum(x_est15);
S15=sum(x15(:,st:en));
% display( strcat(['P_est15 = ',num2str(s1),'  and   p_true= ',num2str(s2)]))
p15=(s15/sum(y(:,1:DAY)))*100;
P15=(S15/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 15 is ',num2str(p15),'% of the whole and for ground truth is  = ',num2str(P15),'%']))
display('#')
N15=norm(x15(:,st:en)-x_est15);

x_est16=x16(:,1:k)*sest(15*k+1:16*k,:);

MAX_X=max(x16(:)); 
MIN_X=min(x16(:));
x_est16(x_est16>MAX_X)=MAX_X;
x_est16(x_est16<MIN_X)=MIN_X;

RMSE16=rmse(x16(:,st:en),x_est16);
display( strcat(['RMSE16=',num2str(RMSE16)]))
s16=sum(x_est16);
S16=sum(x16(:,st:en));
% display( strcat(['P_est16 = ',num2str(s1),'  and   p_true= ',num2str(s2)]))
p16=(s16/sum(y(:,1:DAY)))*100;
P16=(S16/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 16 is ',num2str(p16),'% of the whole and for ground truth is  = ',num2str(P16),'%']))
display('#')
N16=norm(x16(:,st:en)-x_est16);

x_est17=x17(:,1:k)*sest(16*k+1:17*k,:);

MAX_X=max(x17(:)); 
MIN_X=min(x17(:));
x_est17(x_est17>MAX_X)=MAX_X;
x_est17(x_est17<MIN_X)=MIN_X;

RMSE17=rmse(x17(:,st:en),x_est17);
display( strcat(['RMSE17=',num2str(RMSE17)]))
s17=sum(x_est17);
S17=sum(x17(:,st:en));
% display( strcat(['P_est17 = ',num2str(s1),'  and   p_true= ',num2str(s2)]))
p17=(s17/sum(y(:,1:DAY)))*100;
P17=(S17/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 17 is ',num2str(p17),'% of the whole and for ground truth is  = ',num2str(P17),'%']))
display('#')
N17=norm(x17(:,st:en)-x_est17);

x_est18=x18(:,1:k)*sest(17*k+1:18*k,:);

MAX_X=max(x18(:)); 
MIN_X=min(x18(:));
x_est18(x_est18>MAX_X)=MAX_X;
x_est18(x_est18<MIN_X)=MIN_X;

RMSE18=rmse(x18(:,st:en),x_est18);
display( strcat(['RMSE18=',num2str(RMSE18)]))
s18=sum(x_est18);
S18=sum(x18(:,st:en));
% display( strcat(['P_est = ',num2str(s1),'  and   p_true= ',num2str(s2)]))
p18=(s18/sum(y(:,1:DAY)))*100;
P18=(S18/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 18 is ',num2str(p18),'% of the whole and for ground truth is  = ',num2str(P18),'%']))
display('#')

x_est19=x19(:,1:k)*sest(18*k+1:19*k,:);

MAX_X=max(x19(:)); 
MIN_X=min(x19(:));
x_est19(x_est19>MAX_X)=MAX_X;
x_est19(x_est19<MIN_X)=MIN_X;

RMSE19=rmse(x19(:,st:en),x_est19);
display( strcat(['RMSE19=',num2str(RMSE19)]))
s19=sum(x_est19);
S19=sum(x19(:,st:en));
% display( strcat(['P_est19 = ',num2str(s1),'  and   p_true= ',num2str(s2)]))
p19=(s19/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100);
P19=(S19/sum(X(:,1:DAY)))*100;
display( strcat(['Estimated Power usage of device 19 is ',num2str(p19),'% of the whole and for ground truth is  = ',num2str(P19),'%']))
display('#')

RMSE=(RMSE1+RMSE2+RMSE3+RMSE4+RMSE5+RMSE6+RMSE7+RMSE8+RMSE9+RMSE10+RMSE11+RMSE12+RMSE13+RMSE14+RMSE15+RMSE16+RMSE17)/17; % Ignore two last devices ( They are unusual or the same) 
display( strcat(['RMSE=',num2str(RMSE)]))
DIS_ERROR=0.5*(N1+N2+N3+N4+N5+N6+N7+N8+N9+N10+N11+N12+N13+N14+N15+N16+N17);
display( strcat(['Disaggregation Error =',num2str(DIS_ERROR)]))
display( strcat(['RMSE_AGG=',num2str(RMSE_AGG)]))
R=rank(A);
display( strcat(['Rank of dictionary=',num2str(R)]))
%% 
%ploting the pie charts

  T1=s1/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG1=S1/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 1 is ',num2str(T1),'% of the whole and for ground truth is  = ',num2str(TG1),'%']))
display('#')

  T2=s2/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG2=S2/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 2 is ',num2str(T2),'% of the whole and for ground truth is  = ',num2str(TG2),'%']))
display('#')

  T3=s3/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG3=S3/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 3 is ',num2str(T3),'% of the whole and for ground truth is  = ',num2str(TG3),'%']))
display('#')
  T4=s4/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG4=S4/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 4 is ',num2str(T4),'% of the whole and for ground truth is  = ',num2str(TG4),'%']))
display('#')
  T5=s5/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG5=S5/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 5 is ',num2str(T5),'% of the whole and for ground truth is  = ',num2str(TG5),'%']))
display('#')
  T6=s6/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG6=S6/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 6 is ',num2str(T6),'% of the whole and for ground truth is  = ',num2str(TG6),'%']))
display('#')
  T7=s7/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG7=S7/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 7 is ',num2str(T7),'% of the whole and for ground truth is  = ',num2str(TG7),'%']))
display('#')
  T8=s8/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG8=S8/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 8 is ',num2str(T8),'% of the whole and for ground truth is  = ',num2str(TG8),'%']))
display('#')
  T9=s9/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG9=S9/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 9 is ',num2str(T9),'% of the whole and for ground truth is  = ',num2str(TG9),'%']))
display('#')
  T10=s10/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG10=S10/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 10 is ',num2str(T10),'% of the whole and for ground truth is  = ',num2str(TG10),'%']))
display('#')
  T11=s11/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG11=S11/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 11 is ',num2str(T11),'% of the whole and for ground truth is  = ',num2str(TG11),'%']))
display('#')
  T12=s12/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG12=S12/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 12 is ',num2str(T12),'% of the whole and for ground truth is  = ',num2str(TG12),'%']))
display('#')
  T13=s13/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG13=S13/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 13 is ',num2str(T13),'% of the whole and for ground truth is  = ',num2str(TG13),'%']))
display('#')
  T14=s14/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG14=S14/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 14 is ',num2str(T14),'% of the whole and for ground truth is  = ',num2str(TG14),'%']))
display('#')
  T15=s15/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG15=S15/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 15 is ',num2str(T15),'% of the whole and for ground truth is  = ',num2str(TG15),'%']))
display('#')
  T16=s16/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG16=S16/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 16 is ',num2str(T16),'% of the whole and for ground truth is  = ',num2str(TG16),'%']))
display('#')
  T17=s17/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG17=S17/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 17 is ',num2str(T17),'% of the whole and for ground truth is  = ',num2str(TG17),'%']))
display('#')
  T18=s18/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG18=S18/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 18 is ',num2str(T18),'% of the whole and for ground truth is  = ',num2str(TG18),'%']))
display('#')
  T19=s19/sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19)*100;
      TG19=S19/sum(S1+S2+S3+S4+S5+S6+S7+S8+S9+S10+S11+S12+S13+S14+S15+S16+S17+S18+S19)*100;
      display( strcat(['Estimated Power usage of device 19 is ',num2str(T19),'% of the whole and for ground truth is  = ',num2str(TG19),'%']))
display('#')

T=[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19];
TG=[TG1,TG2,TG3,TG4,TG5,TG6,TG7,TG8,TG9,TG10,TG11,TG12,TG13,TG14,TG15,TG16,TG17,TG18,TG19];


est=[s1,s2,s3,s4,s6,s9,s10,s11,s12,s13,s14,s15,s17,s18,s19];
% est_p=[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19];
est_p=[p1,p2,p3,p4,p6,p9,p10,p11,p12,p13,p14,p15,p17,p18,p19]+0.00000001;
gr_truth=[S1,S2,S3,S4,S6,S9,S10,S11,S12,S13,S14,S15,S17,S18,S19]+0.00000001;
% Gr_P=[P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P19];
Gr_P=[P1,P2,P3,P4,P6,P9,P10,P11,P12,P13,P14,P15,P17,P18,P19]+0.00000001;
labels={ '1-North Bedroom','2-Master and South Bedroom', '3-Basement Plugs and Lights', ... 
 '4-Clothes Dryer', '6-Dining Room Plugs', ... 
 '9-Security/Network Equipment', '10-Kitchen Fridge', '11-Air condition ', ...
 '12-Garage', '13-Heat Pump', '14-Instant Hot Water Unit', '15-Home Office','17-Entertainment (TV, PVR, AMP)', '18-Utility Room Plug','19-Wall oven'};

% labels={'1-North Bedroom', '2-Master and South Bedroom', '3-Basement Plugs and Lights', ... 
%  '4-Clothes Dryer','5-Clothes Washer', '6-Dining Room Plugs', ... 
%  '7-Dishwasher', '8-Electronics Workbench','9-Security/Network Equipment', '10-Kitchen Fridge', '11-Air condition ', ...
%  '12-Garage', '13-Heat Pump', '14-Instant Hot Water Unit', '15-Home Office', '16-Outside Plug', '17-Entertainment (TV, PVR, AMP)', '18-Utility Room Plug','19-Wall oven'};

% North Bedroom, Master and South Bedroom, Basement Plugs and Lights, Clothes Dryer,
% Clothes Washer, Dining Room Plugs, Dishwasher, Electronics Workbench, Security/Network Equipment, Kitchen Fridge, Forced Air Furnace (Fan and Thermostat), Garage, Heat Pump, 
% Instant Hot Water Unit, Home Office, Outside Plug, Entertainment (TV, PVR, AMP), and Utility Room Plug

figure;
ax1 = subplot(1,2,1);
pie(ax1,T)
title(ax1,'ESTIMATED');

ax1 = subplot(1,2,2);
pie(ax1,TG)
title(ax1,'GROUND TRUTH');
legend(labels,'Location','southoutside','Orientation','vertical')
%%

%[Xsub,idx]=find_indepenent_columns(A);
display('******************End of results**********************')
%   break;
% pause on;
% pause;
%%
%make it more flexible. en=U 
U=364;
%%
show_all=0; 
if show_all==1
   
figure,     %number of days
subplot(4,2,1),plot(x_est1(:,1:DAY),'linewidth',2),title(strcat(['Estimated signal, "North Bedroom"  RMSE=',num2str(RMSE1)])),ylabel('Current(AMP)');xlabel('Time(MIN)'),
subplot(4,2,2), plot(x1(:,st:en),'linewidth',2),title('True signal,  "North Bedroom" '),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est1(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
subplot(4,2,3),plot(x_est2(:,1:DAY),'linewidth',2),title(strcat(['Master and South bedroom  RMSE=',num2str(RMSE2)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,4), plot(x2(:,st:en),'linewidth',2),title('Master and South bedroom '),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est2(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
subplot(4,2,5),plot(x_est3(:,1:DAY),'linewidth',2),title(strcat(['Basement Plugs and Lights  RMSE=',num2str(RMSE3)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,6), plot(x3(:,st:en),'linewidth',2),title('Basement Plugs and Lights'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est3(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
subplot(4,2,7),plot(x_est4(:,1:DAY),'linewidth',2),title(strcat(['Clothes Dryer  RMSE=',num2str(RMSE4)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,8), plot(x4(:,st:en),'linewidth',2),title('Clothes Dryer'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est4(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
%%%%%%%%%%%%%%
figure, 
subplot(4,2,1),plot(x_est5(:,1:DAY),'linewidth',2),title(strcat(['Estimated signal, Cloths Washer  RMSE=',num2str(RMSE5)])),ylabel('Current(AMP)');xlabel('Time(MIN)'),
subplot(4,2,2), plot(x5(:,st:en),'linewidth',2),title('TRUE SIGNAL,   Cloths Washer'), ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est5(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
subplot(4,2,3),plot(x_est6(:,1:DAY),'linewidth',2),title(strcat(['Dining Room plugs  RMSE=',num2str(RMSE6)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,4), plot(x6(:,st:en),'linewidth',2),title('Dining Room plugs'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est6(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
subplot(4,2,5),plot(x_est7(:,1:DAY),'linewidth',2),title(strcat(['Dishwasher  RMSE=',num2str(RMSE7)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,6), plot(x7(:,st:en),'linewidth',2),title('Dishwasher'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est7(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
subplot(4,2,7),plot(x_est8(:,1:DAY),'linewidth',2),title(strcat(['Electronics Workbench  RMSE=',num2str(RMSE8)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,8), plot(x8(:,st:en),'linewidth',2),title('Electronics Workbench'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est8(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
%%%%%%%%%%%%%%%%%%
figure,
subplot(4,2,1),plot(x_est9(:,1:DAY),'linewidth',2),title(strcat(['Security/Network Equipment  RMSE=',num2str(RMSE9)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,2), plot(x9(:,st:en),'linewidth',2),title('Security/Network'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est9(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
subplot(4,2,3),plot(x_est10(:,1:DAY),'linewidth',2),title(strcat(['Kitchen Fridge  RMSE=',num2str(RMSE10)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,4), plot(x10(:,st:en),'linewidth',2),title('Kitchen Fridge'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est10(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
%based on real numbers 
figure, 
subplot(4,2,1),plot(x_est11(:,1:DAY),'linewidth',2),title(strcat(['Estimated signal, Forced Air Furnace: Fan and Thermostat  RMSE=',num2str(RMSE11)])),ylabel('Current(AMP)');xlabel('Time(MIN)'),
subplot(4,2,2), plot(x11(:,st:en),'linewidth',2),title('TRUE SIGNAL,  Forced Air Furnace'), ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est11(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
subplot(4,2,3),plot(x_est12(:,1:DAY),'linewidth',2),title(strcat(['Garage  RMSE=',num2str(RMSE12)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,4), plot(x12(:,st:en),'linewidth',2),title('Garage'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est12(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
subplot(4,2,5),plot(x_est13(:,1:DAY),'linewidth',2),title(strcat(['Heat Pump  RMSE=',num2str(RMSE13)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,6), plot(x13(:,st:en),'linewidth',2),title('Heat Pump '),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est13(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
subplot(4,2,7),plot(x_est14(:,1:DAY),'linewidth',2),title(strcat(['Instant Hot Water Unit  RMSE=',num2str(RMSE14)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,8), plot(x14(:,st:en),'linewidth',2),title('Instant Hot Water Unit'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est14(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
%%%%%%%%%%%%%%%%%%%%
figure, 
subplot(4,2,1),plot(x_est15(:,1:DAY),'linewidth',2),title(strcat(['Estimated signal, Home Office  RMSE=',num2str(RMSE15)])),ylabel('Current(AMP)');xlabel('Time(MIN)'),
subplot(4,2,2), plot(x15(:,st:en),'linewidth',2),title('TRUE SIGNAL,   Home Office'), ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est15(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
subplot(4,2,3),plot(x_est16(:,1:DAY),'linewidth',2),title(strcat(['Outside Plug  RMSE=',num2str(RMSE16)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,4), plot(x16(:,st:en),'linewidth',2),title('Outside Plug'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est16(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
subplot(4,2,5),plot(x_est17(:,1:DAY),'linewidth',2),title(strcat(['Entertainment: TV, PVR, AMP  RMSE=',num2str(RMSE17)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,6), plot(x17(:,st:en),'linewidth',2),title('Entertainment'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est17(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
subplot(4,2,7),plot(x_est18(:,1:DAY),'linewidth',2),title(strcat(['Utility Room Plug  RMSE=',num2str(RMSE18)])),ylabel('Current(AMP)');xlabel('Time(MIN)');
subplot(4,2,8), plot(x18(:,st:en),'linewidth',2),title('Utility Room Plug'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est18(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

% figure,  ( FOR LAST DEVICE)
% subplot(4,2,1),plot(x_est19(:,1:DAY),'linewidth',2),title(strcat(['Estimated signal, Wall Oven  RMSE=',num2str(RMSE19)])),ylabel('Current(AMP)');xlabel('Time(MIN)'),
% subplot(4,2,2), plot(x19(:,st:en),'linewidth',2),title('TRUE SIGNAL,   Wall Oven '), ylabel('Current(AMP)');xlabel('Time(MIN)');
% hold on; plot(x_est19(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
end
% for choosing specific days to plot. 
% en=st;
% DAY=1;
figure,
subplot(4,2,1), plot(x1(:,st:en),'linewidth',2),title('True signal,  "North Bedroom" '),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est1(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,2), plot(x2(:,st:en),'linewidth',2),title('Master and South bedroom '),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est2(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,3), plot(x3(:,st:en),'linewidth',2),title('Basement Plugs and Lights'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est3(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,4), plot(x4(:,st:en),'linewidth',2),title('Clothes Dryer'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est4(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,5), plot(x5(:,st:en),'linewidth',2),title('TRUE SIGNAL,   Cloths Washer'), ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est5(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,6), plot(x6(:,st:en),'linewidth',2),title('Dining Room plugs'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est6(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,7), plot(x7(:,st:en),'linewidth',2),title('Dishwasher'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est7(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,8), plot(x8(:,st:en),'linewidth',2),title('Electronics Workbench'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est8(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

figure;
subplot(4,2,1), plot(x9(:,st:en),'linewidth',2),title('Security/Network'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est9(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,2), plot(x10(:,st:en),'linewidth',2),title('Kitchen Fridge'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est10(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,3), plot(x11(:,st:en),'linewidth',2),title('TRUE SIGNAL,  Forced Air Furnace'), ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est11(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,4), plot(x12(:,st:en),'linewidth',2),title('Garage'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est12(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,5), plot(x13(:,st:en),'linewidth',2),title('Heat Pump '),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est13(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,6), plot(x14(:,st:en),'linewidth',2),title('Instant Hot Water Unit'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est14(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,7), plot(x15(:,st:en),'linewidth',2),title('TRUE SIGNAL,   Home Office'), ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est15(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,8), plot(x16(:,st:en),'linewidth',2),title('Outside Plug'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est16(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

figure;
subplot(4,2,1), plot(x17(:,st:en),'linewidth',2),title('TV and Entertainment'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est17(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

subplot(4,2,2), plot(x18(:,st:en),'linewidth',2),title('Utility Room Plug'),ylabel('Current(AMP)');xlabel('Time(MIN)');
hold on; plot(x_est18(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
%%
%For paper
        B_result=1;
        if B_result==1
        figure,


        subplot(4,2,1), plot(x3(:,st:en),'linewidth',2),title('Basement Plugs and Lights'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        hold on; plot(x_est3(:,1:DAY),'r','linewidth',2); legend('True','Estimate');


        subplot(4,2,2), plot(x5(:,st:en),'linewidth',2),title('TRUE SIGNAL,   Cloths Washer'), ylabel('Current(AMP)');xlabel('Time(MIN)');
        hold on; plot(x_est5(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

        subplot(4,2,3), plot(x6(:,st:en),'linewidth',2),title('Dining Room plugs'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        hold on; plot(x_est6(:,1:DAY),'r','linewidth',2); legend('True','Estimate');


        subplot(4,2,4), plot(x9(:,st:en),'linewidth',2),title('Security/Network'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        hold on; plot(x_est9(:,1:DAY),'r','linewidth',2); legend('True','Estimate');

        subplot(4,2,5), plot(x10(:,st:en),'linewidth',2),title('Kitchen Fridge'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        hold on; plot(x_est10(:,1:DAY),'r','linewidth',2); legend('True','Estimate');


        subplot(4,2,6), plot(x12(:,st:en),'linewidth',2),title('Garage'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        hold on; plot(x_est12(:,1:DAY),'r','linewidth',2); legend('True','Estimate');


        subplot(4,2,7), plot(x15(:,st:en),'linewidth',2),title('TRUE SIGNAL,   Home Office'), ylabel('Current(AMP)');xlabel('Time(MIN)');
        hold on; plot(x_est15(:,1:DAY),'r','linewidth',2); legend('True','Estimate');


        subplot(4,2,8), plot(x17(:,st:en),'linewidth',2),title('TV and Entertainment'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        hold on; plot(x_est17(:,1:DAY),'r','linewidth',2); legend('True','Estimate');
        end

  C_result=1; %RESULTS For the second revision of the journal paper-DEC 16
        if C_result==1
        figure, 

        subplot(4,3,1), plot(x3(:,st:en),'linewidth',2),title('Basement Plugs and Lights-Ground Truth'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        subplot(4,3,4), plot(x_est3(:,1:DAY),'r','linewidth',2); title('Basement Plugs and Lights-Estimated'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        %subplot(4,3,3), plot(abs(x3(:,st:en)-x_est3(:,1:DAY)),'g','linewidth',2);title('Basement Plugs and Lights-Absolute Error'),ylabel('Current(AMP)');xlabel('Time(MIN)');

        subplot(4,3,2), plot(x5(:,st:en),'linewidth',2),title('Cloths Washer-Ground Truth'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        subplot(4,3,5), plot(x_est5(:,1:DAY),'r','linewidth',2); title('Cloths Washer-Estimated'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        %subplot(4,3,6), plot(abs(x5(:,st:en)-x_est5(:,1:DAY)),'g','linewidth',2);title('Basement Plugs and Lights-Absolute Error'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        
        subplot(4,3,3), plot(x6(:,st:en),'linewidth',2),title('Dining room plugs-Ground Truth'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        subplot(4,3,6), plot(x_est6(:,1:DAY),'r','linewidth',2); title('Dining room plugs-Estimated'),ylabel('Current(AMP)');xlabel('Time(MIN)');
       % subplot(4,3,9), plot(abs(x6(:,st:en)-x_est6(:,1:DAY)),'g','linewidth',2);title('Basement Plugs and Lights-Absolute Error'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        
        subplot(4,3,7), plot(x9(:,st:en),'linewidth',2),title('Security/Network-Ground Truth'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        subplot(4,3,10), plot(x_est9(:,1:DAY),'r','linewidth',2); title('Security/Network-Estimated'),ylabel('Current(AMP)');xlabel('Time(MIN)');
       % subplot(4,3,12), plot(abs(x9(:,st:en)-x_est9(:,1:DAY)),'g','linewidth',2);title('Basement Plugs and Lights-Absolute Error'),ylabel('Current(AMP)');xlabel('Time(MIN)');
       % figure, 
        subplot(4,3,8), plot(x10(:,st:en),'linewidth',2),title('Kitchen Fridge-Ground Truth'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        subplot(4,3,11), plot(x_est10(:,1:DAY),'r','linewidth',2); title('Kitchen Fridge-Estimated'),ylabel('Current(AMP)');xlabel('Time(MIN)');
       % subplot(4,3,12), plot(abs(x10(:,st:en)-x_est10(:,1:DAY)),'g','linewidth',2);title('Basement Plugs and Lights-Absolute Error'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        
        subplot(4,3,9), plot(x12(:,st:en),'linewidth',2),title('Garage-Ground Truth'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        subplot(4,3,12), plot(x_est12(:,1:DAY),'r','linewidth',2); title('Garage-Estimated'),ylabel('Current(AMP)');xlabel('Time(MIN)');
        end
