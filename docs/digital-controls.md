# Digital controls and estimation

Current-loop gains use separate Ld/Lq proportional gains and R-based integral gain. Cross-coupling and back-EMF feedforward use measured current. A circular modulation clamp limits the voltage vector; tracking anti-windup acts on the applied vector.

The Delay object implements an exact integer-sample FIFO. ADC conversion returns quantized values, clipping flags and LSB size. Encoder conversion reports mechanical counts and wrapped electrical angle. Quantization does not make an invalid sensor sample valid.

The BatteryEKF uses only voltage, current and temperature inputs. Its Jacobian includes nonlinear OCV slope, and Joseph-form covariance correction preserves symmetry and positive semidefiniteness. Large normalized innovations are rejected. The initial SOC offset is deliberate.

Main vehicle BMS limits use the physical pack state. The observer is evaluated separately; the repository does not claim closed-loop safety from its SOC estimate. The outlier study applies a declared voltage disturbance, not an invisible ground-truth correction.
