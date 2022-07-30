# Murphy et al. (2022) Computationally efficient framework for diagnosing, understanding, and predicting biphasic population growth.

using Plots
using LinearAlgebra
using NLopt
using .Threads 
using Interpolations
using Distributions
using Roots
using LaTeXStrings

pyplot() # plot options
fnt = Plots.font("sans-serif", 20) # plot options

if isdir("Biphasic_004_2DProliferationAssayBladder")
else
mkdir("Biphasic_004_2DProliferationAssayBladder") # create new folder in current directory
end
filepath_save = [pwd() * "/Biphasic_004_2DProliferationAssayBladder/"] # location to save figures

#######################################################################################


global t1=[0
2
4
6
8
10
12
14
16
18
20
22
24
26
28
30
32
34
36
38
40
42
44
46
48
50
52
54
56
58
60
62
64
66
68
70
72
74
76
78
80
82
84
86
88
90
92
94
96
98
100
102
104
106
108
110
112
114
116
118
120
122
124
126
128
130
132
134
136];
global data=[25.57955
23.32004
23.96439
24.25462
25.9854
25.01853
26.42977
27.1759
25.55973
26.04235
25.83419
24.8944
28.07545
26.08641
26.64501
25.54005
26.28055
25.94203
29.12856
28.04322
29.83151
31.0001
29.98931
29.28883
30.47513
32.83561
33.15372
34.81736
33.21262
35.20492
35.39108
34.64864
35.77974
39.2402
41.46316
42.78649
44.11588
46.58163
43.34664
46.7083
47.23454
51.19459
49.64172
51.8374
61.71023
58.71451
58.49438
58.52035
55.77976
62.08583
61.54935
64.43327
67.30434
75.4136
75.31943
71.87253
76.53957
86.47322
76.67707
85.27595
90.09295
84.4255
84.05376
88.28455
82.80107
86.93713
89.80676
86.9452
90.2028]/100

###
a=zeros(5)
dd=0.001;
rr=0.05;
CC0=0.25;
TT=48;
Var=0.10;
#TH=-0.8212; #80%
#TH=-1.921; #95%
#TH=-3.317; #99%
TH=-5.4138 #99.9

function logistic_delayed(t1,a)
    r=a[1];
    d=a[2]
    K=1;
    C0=a[3];
    TT=a[4];
    dt=maximum(t1)/10000;
    t1mesh=0:dt:TT;
    t2mesh=TT+dt:dt:maximum(t1)+dt;
    tmesh=vcat(t1mesh,t2mesh);
    C=zeros(length(tmesh));
    f1 = (c,d) -> d*c
    f2 = (c,r,K) -> r*c*(1-c/K);
    C[1]=C0;
        for i in 2:length(t1mesh)
        ctemp = C[i-1]+ f1(C[i-1],d)*dt;
        C[i] =C[i-1]+ 0.5*(f1(ctemp,d)+f1(C[i-1],d))*dt;
        end
        for i in length(t1mesh)+1:length(tmesh)
        ctemp = C[i-1]+ f2(C[i-1],r,K)*dt;
        C[i] =C[i-1]+ 0.5*(f2(ctemp,r,K)+f2(C[i-1],r,K))*dt;
        end 
        f=LinearInterpolation(tmesh,C)
        interp=zeros(length(t1))
        interp=f(t1)
    return interp
    end

function error(data,a)
    y=zeros(length(t1))
    y=logistic_delayed(t1,a);
    e=0;
    dist=Normal(0,a[5]);
    e=loglikelihood(dist,data-y) 
    ee=sum(e)
    return ee
end


function fun(a)
return error(data,a)
end


function optimise(fun,θ₀,lb,ub;
    dv = false,
    method = dv ? :LD_LBFGS : :LN_BOBYQA,
)

if dv || String(method)[2] == 'D'
    tomax = fun
else
    tomax = (θ,∂θ) -> fun(θ)
end

opt = Opt(method,length(θ₀))
opt.max_objective = tomax
opt.lower_bounds = lb       # Lower bound
opt.upper_bounds = ub       # Upper bound
opt.local_optimizer = Opt(:LN_NELDERMEAD, length(θ₀))
res = optimize(opt,θ₀)
return res[[2,1]]
end

#######################################################################################
# MLE

θG = [rr,dd,CC0,TT,Var]
r_lb = 0.001
r_ub = 0.2
d_lb = 0.001
d_ub = 0.2
C0_lb = 0.0
C0_ub = 0.5
T_lb = 0.0
T_ub = 96.0
V_lb = 0.001
V_ub = 1.0


lb=[r_lb,d_lb,C0_lb,T_lb,V_lb];
ub=[r_ub,d_ub,C0_ub,T_ub,V_ub];


(xopt,fopt)  = optimise(fun,θG,lb,ub)
global fmle=fopt
global rmle=xopt[1]
global dmle=xopt[2]
global C0mle=xopt[3]
global Tmle=xopt[4]
global Varmle=xopt[5]

ymle = logistic_delayed(t1,xopt);
t1_smooth = LinRange(0,maximum(t1),10001)
ymle_smooth = logistic_delayed(t1_smooth,xopt);
p1=plot(t1_smooth,ymle_smooth,xlab=L"t",ylab=L"C(t)",legend=false,xlims=(0,144),ylims=(0,1),xticks=[0,24,48,72,96,120,144],yticks=[0,0.2,0.4,0.6,0.8,1.0],lw=4,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:orange)
p1=scatter!(t1,data,markersize = 5,markercolor=:black)


savefig(p1,filepath_save[1] * "Figp1.pdf")

#######################################################################################
# Profiling

#Profile rr
nptss=20
rmin=0.02
rmax=0.06
rrange_lower=reverse(LinRange(rmin,rmle,nptss))
rrange_upper=LinRange(rmle + (rmax-rmle)/nptss,rmax,nptss)

nrange_lower=zeros(4,nptss)
llr_lower=zeros(nptss)
nllr_lower=zeros(nptss)
predict_r_lower=zeros(length(t1_smooth),nptss)

nrange_upper=zeros(4,nptss)
llr_upper=zeros(nptss)
nllr_upper=zeros(nptss)
predict_r_upper=zeros(length(t1_smooth),nptss)

# start at mle and increase parameter (upper)
for i in 1:nptss
    function fun1(aa)
        return error(data,[rrange_upper[i],aa[1],aa[2],aa[3],aa[4]])
    end

    lb1=[d_lb,C0_lb,T_lb,V_lb];
    ub1=[d_ub,C0_ub,T_ub,V_ub];
    
    if i==1
        local θG1=[dmle,C0mle,Tmle,Varmle]
    elseif i==2
        # zero order approximation
        local θG1=nrange_upper[:,i-1]
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[dmle,C0mle,Tmle,Varmle]
        end
    elseif i > 2
        # first order approximation
        local θG1= nrange_upper[:,i-1] + ((rrange_upper[i]-rrange_upper[i-1])./(rrange_upper[i-1]-rrange_upper[i-2]))*(nrange_upper[:,i-1]-nrange_upper[:,i-2])
        # if first order approximation is outside of lower bounds or upper bounds use zero order approximation
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=nrange_upper[:,i-1]
        end
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[dmle,C0mle,Tmle,Varmle]
        end
    end

    local (xo,fo)=optimise(fun1,θG1,lb1,ub1)
    nrange_upper[:,i]=xo[:]
    llr_upper[i]=fo[1]
    predict_r_upper[:,i]=logistic_delayed(t1_smooth,[rrange_upper[i],nrange_upper[1,i],nrange_upper[2,i],nrange_upper[3,i],nrange_upper[4,i]])

    if fo > fmle
        global fmle = fo
        global rmle=rrange_upper[i]
        global dmle=nrange_upper[1,i]
        global C0mle=nrange_upper[2,i]
        global Tmle=nrange_upper[3,i]
        global Varmle=nrange_upper[4,i]
    end
end

# start at mle and decrease parameter (lower)
for i in 1:nptss
    function fun1a(aa)
        return error(data,[rrange_lower[i],aa[1],aa[2],aa[3],aa[4]])
    end

    lb1=[d_lb,C0_lb,T_lb,V_lb];
    ub1=[d_ub,C0_ub,T_ub,V_ub];
    
    if i==1
        local θG1=[dmle,C0mle,Tmle,Varmle]
    elseif i==2
        # zero order approximation
        local θG1=nrange_lower[:,i-1]
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[dmle,C0mle,Tmle,Varmle]
        end
    elseif i > 2
        # first order approximation
        local θG1= nrange_lower[:,i-1] + ((rrange_lower[i]-rrange_lower[i-1])./(rrange_lower[i-1]-rrange_lower[i-2]))*(nrange_lower[:,i-1]-nrange_lower[:,i-2])
        # if first order approximation is outside of lower bounds or upper bounds use zero order approximation
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=nrange_lower[:,i-1]
        end
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[dmle,C0mle,Tmle,Varmle]
        end
    end

    local (xo,fo)=optimise(fun1a,θG1,lb1,ub1)
    nrange_lower[:,i]=xo[:]
    llr_lower[i]=fo[1]
    predict_r_lower[:,i]=logistic_delayed(t1_smooth,[rrange_lower[i],nrange_lower[1,i],nrange_lower[2,i],nrange_lower[3,i],nrange_lower[4,i]])

    if fo > fmle
        global fmle = fo
        global rmle=rrange_lower[i]
        global dmle=nrange_lower[1,i]
        global C0mle=nrange_lower[2,i]
        global Tmle=nrange_lower[3,i]
        global Varmle=nrange_lower[4,i]
    end
end

# combine the lower and upper
rrange = [reverse(rrange_lower); rrange_upper]
nrange = [reverse(nrange_lower); nrange_upper ]
llr = [reverse(llr_lower); llr_upper] 
predict_r = [reverse(predict_r_lower,dims=2)  predict_r_upper]
 
nllr=llr.-maximum(llr);

upper_r=zeros(length(t1_smooth))
lower_r=1000*ones(length(t1_smooth))

for i in 1:(nptss*2)
    if nllr[i] >= TH
        for j in 1:length(t1_smooth)
            upper_r[j]=max(predict_r[j,i],upper_r[j])
            lower_r[j]=min(predict_r[j,i],lower_r[j])
        end
    end
end



#Profile dd
dmin=0.0001;
dmax=0.015;
drange_lower=reverse(LinRange(dmin,dmle,nptss))
drange_upper=LinRange(dmle + (dmax-dmle)/nptss,dmax,nptss)

nrange_lower=zeros(4,nptss)
lld_lower=zeros(nptss)
nlld_lower=zeros(nptss)
predict_d_lower=zeros(length(t1_smooth),nptss)

nrange_upper=zeros(4,nptss)
lld_upper=zeros(nptss)
nlld_upper=zeros(nptss)
predict_d_upper=zeros(length(t1_smooth),nptss)

# start at mle and increase parameter (upper)
for i in 1:nptss
    function fun2(aa)
        return error(data,[aa[1],drange_upper[i],aa[2],aa[3],aa[4]])
    end

    lb1=[r_lb,C0_lb,T_lb,V_lb];
    ub1=[r_ub,C0_ub,T_ub,V_ub];
    
    if i==1
        local θG1=[rmle,C0mle,Tmle,Varmle]
    elseif i==2
        # zero order approximation
        local θG1=nrange_upper[:,i-1]
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,C0mle,Tmle,Varmle]
        end
    elseif i > 2
        # first order approximation
        local θG1= nrange_upper[:,i-1] + ((drange_upper[i]-drange_upper[i-1])./(drange_upper[i-1]-drange_upper[i-2]))*(nrange_upper[:,i-1]-nrange_upper[:,i-2])
        # if first order approximation is outside of lower bounds or upper bounds use zero order approximation
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=nrange_upper[:,i-1]
        end
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,C0mle,Tmle,Varmle]
        end
    end

    local (xo,fo)=optimise(fun2,θG1,lb1,ub1)
    nrange_upper[:,i]=xo[:]
    lld_upper[i]=fo[1]
    predict_d_upper[:,i]=logistic_delayed(t1_smooth,[nrange_upper[1,i],drange_upper[i],nrange_upper[2,i],nrange_upper[3,i],nrange_upper[4,i]])

    if fo > fmle
        global fmle = fo
        global rmle=nrange_upper[1,i]
        global dmle=drange_upper[i]
        global C0mle=nrange_upper[2,i]
        global Tmle=nrange_upper[3,i]
        global Varmle=nrange_upper[4,i]
    end
end

# start at mle and decrease parameter (lower)
for i in 1:nptss
    function fun2a(aa)
        return error(data,[aa[1],drange_lower[i],aa[2],aa[3],aa[4]])
    end

    lb1=[r_lb,C0_lb,T_lb,V_lb];
    ub1=[r_ub,C0_ub,T_ub,V_ub];
    
    if i==1
        local θG1=[rmle,C0mle,Tmle,Varmle]
    elseif i==2
        # zero order approximation
        local θG1=nrange_lower[:,i-1]
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,C0mle,Tmle,Varmle]
        end
    elseif i > 2
        # first order approximation
        local θG1= nrange_lower[:,i-1] + ((drange_lower[i]-drange_lower[i-1])./(drange_lower[i-1]-drange_lower[i-2]))*(nrange_lower[:,i-1]-nrange_lower[:,i-2])
        # if first order approximation is outside of lower bounds or upper bounds use zero order approximation
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=nrange_lower[:,i-1]
        end
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,C0mle,Tmle,Varmle]
        end
    end

    local (xo,fo)=optimise(fun2a,θG1,lb1,ub1)
    nrange_lower[:,i]=xo[:]
    lld_lower[i]=fo[1]
    predict_d_lower[:,i]=logistic_delayed(t1_smooth,[nrange_lower[1,i],drange_lower[i],nrange_lower[2,i],nrange_lower[3,i],nrange_lower[4,i]])

    if fo > fmle
        global fmle = fo
        global rmle=nrange_lower[1,i]
        global dmle=drange_lower[i]
        global C0mle=nrange_lower[2,i]
        global Tmle=nrange_lower[3,i]
        global Varmle=nrange_lower[4,i]
    end
end

# combine the lower and upper
drange = [reverse(drange_lower);drange_upper]
nrange = [reverse(nrange_lower); nrange_upper ]
lld = [reverse(lld_lower); lld_upper] 
predict_d = [reverse(predict_d_lower,dims=2) predict_d_upper]

nlld=lld.-maximum(lld);

upper_d=zeros(length(t1_smooth))
lower_d=1000*ones(length(t1_smooth))

for i in 1:(nptss*2)
    if nlld[i] >= TH
        for j in 1:length(t1_smooth)
            upper_d[j]=max(predict_d[j,i],upper_d[j])
            lower_d[j]=min(predict_d[j,i],lower_d[j])
        end
    end
end


#Profile C0
C0min=0.15
C0max=0.30
C0range_lower=reverse(LinRange(C0min,C0mle,nptss))
C0range_upper=LinRange(C0mle + (C0max-C0mle)/nptss,C0max,nptss)

nrange_lower=zeros(4,nptss)
llC0_lower=zeros(nptss)
nllC0_lower=zeros(nptss)
predict_C0_lower=zeros(length(t1_smooth),nptss)

nrange_upper=zeros(4,nptss)
llC0_upper=zeros(nptss)
nllC0_upper=zeros(nptss)
predict_C0_upper=zeros(length(t1_smooth),nptss)

# start at mle and increase parameter (upper)
for i in 1:nptss
    function fun3(aa)
        return error(data,[aa[1],aa[2],C0range_upper[i],aa[3],aa[4]])
    end

    lb1=[r_lb,d_lb,T_lb,V_lb];
    ub1=[r_ub,d_ub,T_ub,V_ub];
    
    if i==1
        local θG1=[rmle,dmle,Tmle,Varmle]
    elseif i==2
        # zero order approximation
        local θG1=nrange_upper[:,i-1]
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,dmle,Tmle,Varmle]
        end
    elseif i > 2
        # first order approximation
        local θG1= nrange_upper[:,i-1] + ((C0range_upper[i]-C0range_upper[i-1])./(C0range_upper[i-1]-C0range_upper[i-2]))*(nrange_upper[:,i-1]-nrange_upper[:,i-2])
        # if first order approximation is outside of lower bounds or upper bounds use zero order approximation
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=nrange_upper[:,i-1]
        end
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,dmle,Tmle,Varmle]
        end
    end

    local (xo,fo)=optimise(fun3,θG1,lb1,ub1)
    nrange_upper[:,i]=xo[:]
    llC0_upper[i]=fo[1]
    predict_C0_upper[:,i]=logistic_delayed(t1_smooth,[nrange_upper[1,i],nrange_upper[2,i],C0range_upper[i],nrange_upper[3,i],nrange_upper[4,i]])

    if fo > fmle
        global fmle = fo
        global rmle=nrange_upper[1,i]
        global dmle=nrange_upper[2,i]
        global C0mle=C0range_upper[i]
        global Tmle=nrange_upper[3,i]
        global Varmle=nrange_upper[4,i]
    end
end

# start at mle and decrease parameter (lower)
for i in 1:nptss
    function fun3a(aa)
        return error(data,[aa[1],aa[2],C0range_lower[i],aa[3],aa[4]])
    end

    lb1=[r_lb,d_lb,T_lb,V_lb];
    ub1=[r_ub,d_ub,T_ub,V_ub];
    
    if i==1
        local θG1=[rmle,dmle,Tmle,Varmle]
    elseif i==2
        # zero order approximation
        local θG1=nrange_lower[:,i-1]
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,dmle,Tmle,Varmle]
        end
    elseif i > 2
        # first order approximation
        local θG1= nrange_lower[:,i-1] + ((C0range_lower[i]-C0range_lower[i-1])./(C0range_lower[i-1]-C0range_lower[i-2]))*(nrange_lower[:,i-1]-nrange_lower[:,i-2])
        # if first order approximation is outside of lower bounds or upper bounds use zero order approximation
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=nrange_lower[:,i-1]
        end
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,dmle,Tmle,Varmle]
        end
    end

    local (xo,fo)=optimise(fun3a,θG1,lb1,ub1)
    nrange_lower[:,i]=xo[:]
    llC0_lower[i]=fo[1]
    predict_C0_lower[:,i]=logistic_delayed(t1_smooth,[nrange_lower[1,i],nrange_lower[2,i],C0range_lower[i],nrange_lower[3,i],nrange_lower[4,i]])

    if fo > fmle
        global fmle = fo
        global rmle=nrange_lower[1,i]
        global dmle=nrange_lower[2,i]
        global C0mle=C0range_lower[i]
        global Tmle=nrange_lower[3,i]
        global Varmle=nrange_lower[4,i]
    end
end

# combine the lower and upper
C0range = [reverse(C0range_lower);C0range_upper]
nrange = [reverse(nrange_lower); nrange_upper ]
llC0 = [reverse(llC0_lower); llC0_upper] 
predict_C0 = [reverse(predict_C0_lower,dims=2) predict_C0_upper]

nllC0=llC0.-maximum(llC0)

upper_C0=zeros(length(t1_smooth))
lower_C0=1000*ones(length(t1_smooth))

for i in 1:(nptss*2)
    if nllC0[i] >= TH
        for j in 1:length(t1_smooth)
            upper_C0[j]=max(predict_C0[j,i],upper_C0[j])
            lower_C0[j]=min(predict_C0[j,i],lower_C0[j])
        end
    end
end



#Profile T
Tmin=40
Tmax=100
Trange_lower=reverse(LinRange(Tmin,Tmle,nptss))
Trange_upper=LinRange(Tmle + (Tmax-Tmle)/nptss,Tmax,nptss)

nrange_lower=zeros(4,nptss)
llT_lower=zeros(nptss)
nllT_lower=zeros(nptss)
predict_T_lower=zeros(length(t1_smooth),nptss)

nrange_upper=zeros(4,nptss)
llT_upper=zeros(nptss)
nllT_upper=zeros(nptss)
predict_T_upper=zeros(length(t1_smooth),nptss)

# start at mle and increase parameter (upper)
for i in 1:nptss
    function fun4(aa)
        return error(data,[aa[1],aa[2],aa[3],Trange_upper[i],aa[4]])
    end

    lb1=[r_lb,d_lb,C0_lb,V_lb];
    ub1=[r_ub,d_ub,C0_ub,V_ub];
    
    if i==1
        local θG1=[rmle,dmle,C0mle,Varmle]
    elseif i==2
        # zero order approximation
        local θG1=nrange_upper[:,i-1]
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,dmle,C0mle,Varmle]
        end
    elseif i > 2
        # first order approximation
        local θG1= nrange_upper[:,i-1] + ((Trange_upper[i]-Trange_upper[i-1])./(Trange_upper[i-1]-Trange_upper[i-2]))*(nrange_upper[:,i-1]-nrange_upper[:,i-2])
        # if first order approximation is outside of lower bounds or upper bounds use zero order approximation
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=nrange_upper[:,i-1]
        end
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,dmle,C0mle,Varmle]
        end
    end

    local (xo,fo)=optimise(fun4,θG1,lb1,ub1)
    nrange_upper[:,i]=xo[:]
    llT_upper[i]=fo[1]
    predict_T_upper[:,i]=logistic_delayed(t1_smooth,[nrange_upper[1,i],nrange_upper[2,i],nrange_upper[3,i],Trange_upper[i],nrange_upper[4,i]])

    if fo > fmle
        global fmle = fo
        global rmle=nrange_upper[1,i]
        global dmle=nrange_upper[2,i]
        global C0mle=nrange_upper[3,i]
        global Tmle=Trange_upper[i]
        global Varmle=nrange_upper[4,i]
    end
end

# start at mle and decrease parameter (lower)
for i in 1:nptss
    function fun4a(aa)
        return error(data,[aa[1],aa[2],aa[3],Trange_lower[i],aa[4]])
    end

    lb1=[r_lb,d_lb,C0_lb,V_lb];
    ub1=[r_ub,d_ub,C0_ub,V_ub];
    
    if i==1
        local θG1=[rmle,dmle,C0mle,Varmle]
    elseif i==2
        # zero order approximation
        local θG1=nrange_lower[:,i-1]
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,dmle,C0mle,Varmle]
        end
    elseif i > 2
        # first order approximation
        local θG1= nrange_lower[:,i-1] + ((Trange_lower[i]-Trange_lower[i-1])./(Trange_lower[i-1]-Trange_lower[i-2]))*(nrange_lower[:,i-1]-nrange_lower[:,i-2])
        # if first order approximation is outside of lower bounds or upper bounds use zero order approximation
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=nrange_lower[:,i-1]
        end
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,dmle,C0mle,Varmle]
        end
    end

    local (xo,fo)=optimise(fun4a,θG1,lb1,ub1)
    nrange_lower[:,i]=xo[:]
    llT_lower[i]=fo[1]
    predict_T_lower[:,i]=logistic_delayed(t1_smooth,[nrange_lower[1,i],nrange_lower[2,i],nrange_lower[3,i],Trange_lower[i],nrange_lower[4,i]])

    if fo > fmle
        global fmle = fo
        global rmle=nrange_lower[1,i]
        global dmle=nrange_lower[2,i]
        global C0mle=nrange_lower[3,i]
        global Tmle=Trange_lower[i]
        global Varmle=nrange_lower[4,i]
    end
end

# combine the lower and upper
Trange = [reverse(Trange_lower);Trange_upper]
llT = [reverse(llT_lower); llT_upper] 
predict_T = [reverse(predict_T_lower,dims=2) predict_T_upper]

nllT=llT.-maximum(llT);

upper_T=zeros(length(t1_smooth))
lower_T=1000*ones(length(t1_smooth))

for i in 1:(nptss*2)
    if nllT[i] >= TH
        for j in 1:length(t1_smooth)
            upper_T[j]=max(predict_T[j,i],upper_T[j])
            lower_T[j]=min(predict_T[j,i],lower_T[j])
        end
    end
end



#Profile Var
Vmin=0.0001
Vmax=0.05
Vrange_lower=reverse(LinRange(Vmin,Varmle,nptss))
Vrange_upper=LinRange(Varmle + (Vmax-Varmle)/nptss,Vmax,nptss)

nrange_lower=zeros(4,nptss)
llV_lower=zeros(nptss)
nllV_lower=zeros(nptss)
predict_V_lower=zeros(length(t1_smooth),nptss)

nrange_upper=zeros(4,nptss)
llV_upper=zeros(nptss)
nllV_upper=zeros(nptss)
predict_V_upper=zeros(length(t1_smooth),nptss)

# start at mle and increase parameter (upper)
for i in 1:nptss
    function fun5(aa)
        return error(data,[aa[1],aa[2],aa[3],aa[4],Vrange_upper[i],])
    end

    lb1=[r_lb,d_lb,C0_lb,T_lb];
    ub1=[r_ub,d_ub,C0_ub,T_ub];

    if i==1
        local θG1=[rmle,dmle,C0mle,Tmle]
    elseif i==2
        # zero order approximation
        local θG1=nrange_upper[:,i-1]
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,dmle,C0mle,Tmle]
        end
    elseif i > 2
        # first order approximation
        local θG1= nrange_upper[:,i-1] + ((Vrange_upper[i]-Vrange_upper[i-1])./(Vrange_upper[i-1]-Vrange_upper[i-2]))*(nrange_upper[:,i-1]-nrange_upper[:,i-2])
        # if first order approximation is outside of lower bounds or upper bounds use zero order approximation
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=nrange_upper[:,i-1]
        end
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,dmle,C0mle,Tmle]
        end
    end

    local (xo,fo)=optimise(fun5,θG1,lb1,ub1)
    nrange_upper[:,i]=xo[:]
    llV_upper[i]=fo[1]
    predict_V_upper[:,i]=logistic_delayed(t1_smooth,[nrange_upper[1,i],nrange_upper[2,i],nrange_upper[3,i],nrange_upper[4,i],Vrange_upper[i],])

    if fo > fmle
        global fmle = fo
        global rmle=nrange_upper[1,i]
        global dmle=nrange_upper[2,i]
        global C0mle=nrange_upper[3,i]
        global Tmle=nrange_upper[4,i]
        global Varmle=Vrange_upper[i]
    end
end

# start at mle and decrease parameter (lower)
for i in 1:nptss
    function fun5a(aa)
        return error(data,[aa[1],aa[2],aa[3],aa[4],Vrange_lower[i],])
    end


    lb1=[r_lb,d_lb,C0_lb,T_lb];
    ub1=[r_ub,d_ub,C0_ub,T_ub];
    
    if i==1
        local θG1=[rmle,dmle,C0mle,Tmle]
    elseif i==2
        # zero order approximation
        local θG1=nrange_lower[:,i-1]
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,dmle,C0mle,Tmle]
        end
    elseif i > 2
        # first order approximation
        local θG1= nrange_lower[:,i-1] + ((Vrange_lower[i]-Vrange_lower[i-1])./(Vrange_lower[i-1]-Vrange_lower[i-2]))*(nrange_lower[:,i-1]-nrange_lower[:,i-2])
        # if first order approximation is outside of lower bounds or upper bounds use zero order approximation
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=nrange_lower[:,i-1]
        end
        # if zero order approximation outside of bounds use MLE
        if (sum(θG1.< lb1) + sum(θG1 .> ub1)) > 0
            local θG1=[rmle,dmle,C0mle,Tmle]
        end
    end

    local (xo,fo)=optimise(fun5a,θG1,lb1,ub1)
    nrange_lower[:,i]=xo[:]
    llV_lower[i]=fo[1]
    predict_V_lower[:,i]=logistic_delayed(t1_smooth,[nrange_lower[1,i],nrange_lower[2,i],nrange_lower[3,i],nrange_lower[4,i],Vrange_lower[i],])

    if fo > fmle
        global fmle = fo
        global rmle=nrange_lower[1,i]
        global dmle=nrange_lower[2,i]
        global C0mle=nrange_lower[3,i]
        global Tmle=nrange_lower[4,i]
        global Varmle=Vrange_lower[i]
    end
end

# combine the lower and upper
Vrange = [reverse(Vrange_lower);Vrange_upper]
nrange = [reverse(nrange_lower); nrange_upper ]
llV = [reverse(llV_lower); llV_upper] 
predict_V = [reverse(predict_V_lower,dims=2) predict_V_upper]

nllV=llV.-maximum(llV);

upper_V=zeros(length(t1_smooth))
lower_V=1000*ones(length(t1_smooth))

for i in 1:(nptss*2)
    if nllV[i] >= TH
        for j in 1:length(t1_smooth)
            upper_V[j]=max(predict_V[j,i],upper_V[j])
            lower_V[j]=min(predict_V[j,i],lower_V[j])
        end
    end
end


# interpolate for smoother profile likelihoods
interp_nptss= 1001;

# r
interp_points_rrange =  LinRange(rmin,rmax,interp_nptss)
interp_r = LinearInterpolation(rrange,nllr)
interp_nllr = interp_r(interp_points_rrange)

s1=plot(interp_points_rrange,interp_nllr,xlim=(rmin,rmax),ylim=(-6,0.1),xticks=[0.02,0.04,0.06],yticks=[-5,-4,-3,-2,-1,0],xlab=L"r_{2}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
s1=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
s1=hline!([-3.317],lw=2,linecolor=:black,linestyle=:dash)
s1=hline!([-5.4138],lw=2,linecolor=:black,linestyle=:dashdot)
s1=vline!([rmle],lw=3,linecolor=:red)

# d
interp_points_drange =  LinRange(dmin,dmax,interp_nptss)
interp_d = LinearInterpolation(drange,nlld)
interp_nlld = interp_d(interp_points_drange)

s2=plot(interp_points_drange,interp_nlld,xlim=(0,dmax),ylim=(-6,0.1),xticks=[0,0.005,0.010,0.015],yticks=[-5,-4,-3,-2,-1,0],xlab=L"r_{1}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
s3=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
s3=hline!([-3.317],lw=2,linecolor=:black,linestyle=:dash)
s3=hline!([-5.4138],lw=2,linecolor=:black,linestyle=:dashdot)
s3=vline!([dmle],lw=3,linecolor=:red)

# C0
interp_points_C0range =  LinRange(C0min,C0max,interp_nptss)
interp_C0 = LinearInterpolation(C0range,nllC0)
interp_nllC0 = interp_C0(interp_points_C0range)

s3=plot(interp_points_C0range,interp_nllC0,xlim=(C0min,C0max),ylim=(-6,0.1),xticks=[0.15,0.20,0.25,0.30],yticks=[-5,-4,-3,-2,-1,0],xlab=L"C(0)",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
s3=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
s3=hline!([-3.317],lw=2,linecolor=:black,linestyle=:dash)
s3=hline!([-5.4138],lw=2,linecolor=:black,linestyle=:dashdot)
s3=vline!([C0mle],lw=3,linecolor=:red)


# T
interp_points_Trange =  LinRange(Tmin,Tmax,interp_nptss)
interp_T = LinearInterpolation(Trange,nllT)
interp_nllT = interp_T(interp_points_Trange)

s4=plot(interp_points_Trange,interp_nllT,xlim=(Tmin,Tmax),ylim=(-6,0.1),xticks=[40,60,80,100],yticks=[-5,-4,-3,-2,-1,0],xlab=L"T",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
s4=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
s4=hline!([-3.317],lw=2,linecolor=:black,linestyle=:dash)
s4=hline!([-5.4138],lw=2,linecolor=:black,linestyle=:dashdot)
s4=vline!([Tmle],lw=3,linecolor=:red)

# Var
interp_points_Vrange =  LinRange(Vmin,Vmax,interp_nptss)
interp_V = LinearInterpolation(Vrange,nllV)
interp_nllV = interp_V(interp_points_Vrange)

s5=plot(interp_points_Vrange,interp_nllV,xlim=(Vmin,Vmax),ylim=(-6,0.1),yticks=[-5,-4,-3,-2,-1,0],xlab=L"\sigma^{2}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
s5=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
s5=hline!([-3.317],lw=2,linecolor=:black,linestyle=:dash)
s5=hline!([-5.4138],lw=2,linecolor=:black,linestyle=:dashdot)
s5=vline!([Varmle],lw=3,linecolor=:red)

# Combined
s6=plot(p1,s1,s2,s3,s4,s5,layout=(2,3),legend=false)
display(s6)

# Recompute best fit

t1_smooth = LinRange(0,maximum(t1),10001)
ymle_smooth = logistic_delayed(t1_smooth,[rmle;dmle;C0mle;Tmle;Varmle]);
p1_updated=plot(t1_smooth,ymle_smooth,xlab=L"t",ylab=L"C(t)",legend=false,xlims=(0,144),ylims=(0,1),xticks=[0,24,48,72,96,120,144],yticks=[0,0.2,0.4,0.6,0.8,1.0],lw=4,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:orange)
p1_updated=scatter!(t1,data,markersize = 5,markercolor=:black)

display(p1_updated)
savefig(p1_updated,filepath_save[1] * "Figp1_updated.pdf")

# save figures

display(p1)
savefig(p1,filepath_save[1] * "Figp1.pdf")

display(s1)
savefig(s1,filepath_save[1] * "Figs1.pdf")

display(s2)
savefig(s2,filepath_save[1] * "Figs2.pdf")

display(s3)
savefig(s3,filepath_save[1] * "Figs3.pdf")

display(s4)
savefig(s4,filepath_save[1] * "Figs4.pdf")

display(s5)
savefig(s5,filepath_save[1] * "Figs5.pdf")

display(s6)
savefig(s6,filepath_save[1] * "Figs6.pdf")


#######################################################################################
# Parameter-wise profile predictions


upper=zeros(length(t1_smooth))
lower=zeros(length(t1_smooth))
for i in 1:length(t1_smooth)
    upper[i] = max(upper_r[i],upper_d[i],upper_C0[i],upper_V[i],upper_T[i])
    lower[i] = min(lower_r[i],lower_d[i],lower_C0[i],lower_V[i],lower_T[i])
end


q2=plot(t1_smooth,ymle_smooth,w=1,c=:red)
q2=plot!(t1_smooth,ymle_smooth,w=0,c=:blue,ribbon=(ymle_smooth.-lower_r,upper_r.-ymle_smooth),fillalpha=.2,xlim=(0,144),ylim=(0,1),xticks=[0,24,48,72,96,120,144],legend=false,xlab=L"t",ylab=L"C(t)",title=L"r_{2}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)

q3=plot(t1_smooth,ymle_smooth,w=1,c=:red)
q3=plot!(t1_smooth,ymle_smooth,w=0,c=:blue,ribbon=(ymle_smooth.-lower_d,upper_d.-ymle_smooth),fillalpha=.2,xlim=(0,144),ylim=(0,1),xticks=[0,24,48,72,96,120,144],legend=false,xlab=L"t",ylab=L"C(t)",title=L"r_{1}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)

q4=plot(t1_smooth,ymle_smooth,w=1,c=:red)
q4=plot!(t1_smooth,ymle_smooth,w=0,c=:blue,ribbon=(ymle_smooth.-lower_C0,upper_C0.-ymle_smooth),fillalpha=.2,xlim=(0,144),ylim=(0,1),xticks=[0,24,48,72,96,120,144],legend=false,xlab=L"t",ylab=L"C(t)",title=L"C(0)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)

q5=plot(t1_smooth,ymle_smooth,w=1,c=:red)
q5=plot!(t1_smooth,ymle_smooth,w=0,c=:blue,ribbon=(ymle_smooth.-lower_T,upper_T.-ymle_smooth),fillalpha=.2,xlim=(0,144),ylim=(0,1),xticks=[0,24,48,72,96,120,144],legend=false,xlab=L"t",ylab=L"C(t)",title=L"T",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)

q6=plot(t1_smooth,ymle_smooth,w=1,c=:red)
q6=plot!(t1_smooth,ymle_smooth,w=0,c=:blue,ribbon=(ymle_smooth.-lower_V,upper_V.-ymle_smooth),fillalpha=.2,xlim=(0,144),ylim=(0,1),xticks=[0,24,48,72,96,120,144],legend=false,xlab=L"t",ylab=L"C(t)",title=L"\sigma^{2}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)

q7=plot(t1_smooth,ymle_smooth,w=1,c=:red)
q7=plot!(t1_smooth,ymle_smooth,w=0,c=:blue,ribbon=(ymle_smooth.-lower,upper.-ymle_smooth),fillalpha=.2,xlim=(0,144),ylim=(0,1),xticks=[0,24,48,72,96,120,144],legend=false,xlab=L"t",ylab=L"C(t)",title=L"\mathrm{Union}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)


qq1=plot(q2,q3,q4,q5,q6,q7,layout=(2,3),legend=false)
display(qq1)


display(q2)
savefig(q2,filepath_save[1] * "Figq2.pdf")

display(q3)
savefig(q3,filepath_save[1] * "Figq3.pdf")

display(q4)
savefig(q4,filepath_save[1] * "Figq4.pdf")

display(q5)
savefig(q5,filepath_save[1] * "Figq5.pdf")

display(q6)
savefig(q6,filepath_save[1] * "Figq6.pdf")

display(q7)
savefig(q7,filepath_save[1] * "Figq7.pdf")


display(qq1)
savefig(qq1,filepath_save[1] * "Figqq1.pdf")


# Parameter-wise profile predictions - difference from mle

y_zero = 0*ones(length(t1_smooth),1);



qdiff2=plot(t1_smooth,y_zero,w=1,c=:red)
qdiff2=plot!(t1_smooth,y_zero,w=0,c=:blue,ribbon=(ymle_smooth.-lower_r,upper_r.-ymle_smooth),fillalpha=.2,xlim=(0,144),ylim=(-0.1,0.1),xticks=[0,24,48,72,96,120,144],legend=false,xlab=L"t",ylab=L"\mathcal{I}_{q}^{r_{2}} - y(\hat{\theta})",title=L"r_{2}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)

qdiff3=plot(t1_smooth,y_zero,w=1,c=:red)
qdiff3=plot!(t1_smooth,y_zero,w=0,c=:blue,ribbon=(ymle_smooth.-lower_d,upper_d.-ymle_smooth),fillalpha=.2,xlim=(0,144),ylim=(-0.1,0.1),xticks=[0,24,48,72,96,120,144],legend=false,xlab=L"t",ylab=L"\mathcal{I}_{q}^{r_{1}} - y(\hat{\theta})",title=L"r_{1}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)

qdiff4=plot(t1_smooth,y_zero,w=1,c=:red)
qdiff4=plot!(t1_smooth,y_zero,w=0,c=:blue,ribbon=(ymle_smooth.-lower_C0,upper_C0.-ymle_smooth),fillalpha=.2,xlim=(0,144),ylim=(-0.1,0.1),xticks=[0,24,48,72,96,120,144],legend=false,xlab=L"t",ylab=L"\mathcal{I}_{q}^{C(0)} - y(\hat{\theta})",title=L"C(0)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)

qdiff5=plot(t1_smooth,y_zero,w=1,c=:red)
qdiff5=plot!(t1_smooth,y_zero,w=0,c=:blue,ribbon=(ymle_smooth.-lower_T,upper_T.-ymle_smooth),fillalpha=.2,xlim=(0,144),ylim=(-0.1,0.1),xticks=[0,24,48,72,96,120,144],legend=false,xlab=L"t",ylab=L"\mathcal{I}_{q}^{T} - y(\hat{\theta})",title=L"T",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)

qdiff6=plot(t1_smooth,y_zero,w=1,c=:red)
qdiff6=plot!(t1_smooth,y_zero,w=0,c=:blue,ribbon=(ymle_smooth.-lower_V,upper_V.-ymle_smooth),fillalpha=.2,xlim=(0,144),ylim=(-0.1,0.1),xticks=[0,24,48,72,96,120,144],legend=false,xlab=L"t",ylab=L"\mathcal{I}_{q}^{\sigma^{2}} - y(\hat{\theta})",title=L"\sigma^{2}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)


qdiff7=plot(t1_smooth,y_zero,w=1,c=:red)
qdiff7=plot!(t1_smooth,y_zero,w=0,c=:blue,ribbon=(ymle_smooth.-lower,upper.-ymle_smooth),fillalpha=.2,xlim=(0,144),ylim=(-0.1,0.1),xticks=[0,24,48,72,96,120,144],legend=false,xlab=L"t",ylab=L"\mathcal{I}_{q}^{T, C(0), r_{1}, r_{2}, \sigma^{2}}- y(\hat{\theta})",title=L"\mathrm{Union}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)


qdiffq1=plot(qdiff2,qdiff3,qdiff4,qdiff5,qdiff6,qdiff7,layout=(2,3),legend=false)

# save figures

display(qdiff2)
savefig(qdiff2,filepath_save[1] * "Figqdiff2.pdf")

display(qdiff3)
savefig(qdiff3,filepath_save[1] * "Figqdiff3.pdf")

display(qdiff4)
savefig(qdiff4,filepath_save[1] * "Figqdiff4.pdf")

display(qdiff5)
savefig(qdiff5,filepath_save[1] * "Figqdiff5.pdf")

display(qdiff6)
savefig(qdiff6,filepath_save[1] * "Figqdiff6.pdf")

display(qdiff7)
savefig(qdiff7,filepath_save[1] * "Figqdiff7.pdf")


display(qdiffq1)
savefig(qdiffq1,filepath_save[1] * "Figqdiffq1.pdf")

#######################################################################################
# compute the bounds of confidence interval

function fun_interpCI(mle,interp_points_range,interp_nll,TH)
    # find bounds of CI

    range_minus_mle = interp_points_range - mle*ones(length(interp_points_range),1)
    abs_range_minus_mle = broadcast(abs, range_minus_mle)
    findmin_mle = findmin(abs_range_minus_mle)

    # find closest value to CI threshold intercept
    value_minus_threshold = interp_nll - TH*ones(length(interp_nllr),1)
    abs_value_minus_threshold = broadcast(abs, value_minus_threshold)
    lb_CI_tmp = findmin(abs_value_minus_threshold[1:findmin_mle[2][1]])
    ub_CI_tmp = findmin(abs_value_minus_threshold[findmin_mle[2][1]:length(abs_value_minus_threshold)])
    lb_CI = interp_points_range[lb_CI_tmp[2][1]]
    ub_CI = interp_points_range[findmin_mle[2][1]-1 + ub_CI_tmp[2][1]]

    return lb_CI,ub_CI
end

# r (r_2)
(lb_CI_r,ub_CI_r) = fun_interpCI(rmle,interp_points_rrange,interp_nllr,TH)
println(round(lb_CI_r; digits = 4))
println(round(ub_CI_r; digits = 4))

# d (r_1)
(lb_CI_d,ub_CI_d) = fun_interpCI(dmle,interp_points_drange,interp_nlld,TH)
println(round(lb_CI_d; digits = 3))
println(round(ub_CI_d; digits = 3))

# C0
(lb_CI_C0,ub_CI_C0) = fun_interpCI(C0mle,interp_points_C0range,interp_nllC0,TH)
println(round(lb_CI_C0; digits = 3))
println(round(ub_CI_C0; digits = 3))

# T
(lb_CI_T,ub_CI_T) = fun_interpCI(Tmle,interp_points_Trange,interp_nllT,TH)
println(round(lb_CI_T; digits = 3))
println(round(ub_CI_T; digits = 3))

# V
(lb_CI_V,ub_CI_V) = fun_interpCI(Varmle,interp_points_Vrange,interp_nllV,TH)
println(round(lb_CI_V; digits = 3))
println(round(ub_CI_V; digits = 3))

