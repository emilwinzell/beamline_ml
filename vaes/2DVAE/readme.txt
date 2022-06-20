The final VAE, presented in the project.
Trained for 200 epochs on set 05081047 (with bigger limits on the mirror parameters pitch=+/- 1 mRad, yaw = +/- 20 mRad, roll = +/- 20 mRad, lat_transl = +/- 5 mm, vert_transl = +/- 5 mm)
Then retrained for 300 epochs on set 05120912 (with smaller limits: pitch=+/- 3 mRad, yaw = +/- 1 mRad, roll = +/- 1 mRad, lat_transl = +/- 1.5 mm, vert_transl = +/- 2.5 mm)

see report "X-ray beamline alignment using machine learning at MAX IV" for details about implementation and architecture