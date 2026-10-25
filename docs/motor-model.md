# PMSM equations and conventions

Clarke and Park transforms are amplitude invariant. d-q quantities are phase-peak amplitudes; electrical power is 1.5 (vd id + vq iq). Mechanical speed is rad/s; electrical speed is pole-pairs times mechanical speed.

vd = R id + Ld did/dt - we Lq iq.
vq = R iq + Lq diq/dt + we (Ld id + psi).
Torque = 1.5 p [psi + (Ld-Lq) id] iq.

The instantaneous audit separates copper, magnetic storage, and conversion power. Frozen-speed electrical steps use an augmented matrix exponential. Coupled electrical/mechanical steps use RK4 and retain mechanical angle. These are time-domain models; no switched-inverter claim is made for the route replay.

MTPA is solved independently using bounded scalar optimization and a stationary equation. Field weakening enumerates polynomial constraint intersections on a fixed-torque hyperbola, then checks torque/current/voltage residuals. Infeasible torque is reduced by bracketing, and an impossible no-load voltage point opens the drive.

References: [MathWorks constraint curves](https://www.mathworks.com/help/mcb/gs/pmsm-constraint-curves-and-their-application.html), [MTPA](https://www.mathworks.com/help/mcb/ref/mtpacontrolreference.html). Code is independently implemented; no measured-machine validation is claimed.
