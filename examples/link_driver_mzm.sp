* One file, both domains: a CMOS driver feeding a Mach-Zehnder modulator.

.electronic
.model nch NMOS (LEVEL=1 VTO=0.7  KP=120u LAMBDA=0.02)
.model pch PMOS (LEVEL=1 VTO=-0.7 KP=40u  LAMBDA=0.02)
Vdd vdd 0 3.0
Vin g   0 PULSE(0 3 1n 0.2n 0.2n 10n 20n)
MN1 vdrv g 0   0   nch W=8u  L=0.5u
MP1 vdrv g vdd vdd pch W=16u L=0.5u
Rload1 vo1 0 1k
Rload2 vo2 0 1k
.sensparam MN1:W MN1:L
.tran 0 4e-8 40

.photonic
.mode neff=2.35 ng=4.0 wl=1550e-9
.freq 193.1e12 193.1e12 1
.source 1.0@a1 0.0@a2
mzm0 a1 a2 b1 b2 vdrv level3 vpi=2.0 vbias=0.0 il=0.0
pd1 b1 vo1 level1 r0=1.0
pd2 b2 vo2 level1 r0=1.0
