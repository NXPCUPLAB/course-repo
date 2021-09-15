m = 1000;
b = 50;
r = 10;

s = tf('s');
P_cruise = 1/(m*s+b);

rlocus(P_cruise)
axis([-0.6 0 -0.6 0.6]);
sgrid(0.6, 0.36)

%%
%Place the root locus where you desire:
rlocus(P_cruise)
axis([-0.6 0 -0.6 0.6]);
sgrid(0.6, 0.36)

[Kp,poles]=rlocfind(P_cruise)

%% 
%After completing the previous tasks, and getting the value of Kp for your chosen root locus, enter the obtained value of Kp :

Kp = 0; %Enter the value of the obtained Kp.
sys_cl = feedback(Kp*P_cruise,1);
t = 0:0.1:20;
step(r*sys_cl,t)

%%
%Now to add the lag controller (continue with the previous step):
zo = 0.3;
po = 0.03;

s = tf('s');
C_lag = (s+zo)/(s+po);

rlocus(C_lag*P_cruise);
axis([-0.6 0 -0.4 0.4])
sgrid(0.6,0.36);

%%
%Place the root locus where you desire:
rlocus(C_lag*P_cruise);
axis([-0.6 0 -0.4 0.4])
sgrid(0.6,0.36);
[Kp,poles]=rlocfind(C_lag*P_cruise)

%%
%After completing the previous tasks, and getting the value of Kp for your
%chosen root locus, insert the value of Kp and run the code to check the result:

Kp = 0;%Enter the value of the obtained Kp.
sys_cl = feedback(Kp*C_lag*P_cruise,1);
t = 0:0.1:20;
step(r*sys_cl,t)
axis([0 20 0 12])