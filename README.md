# Non-Intrusive Load Monitoring #
## HVAC_ORNL_UTK(HVORUT) dataset: ##
### A dataset for studying NILM for HVAC units###
![LD](/LD.png)

This repository includes our collected dataset for HVAC component energy disaggregation as well as 
the source code and demo for our paper in IEEE Transactions on Power Systems: [Non-Intrusive Energy Disaggregation Using Non-negative Matrix Factorization with Sum-to-k Constraint.](http://ieeexplore.ieee.org/abstract/document/7835299/)
To the best of our knowledge this is the first collected dataset for studying NILM for Heating, Ventilation, and Air Conditioning (HVAC) systems. 

Energy disaggregation or Non-Intrusive Load Monitoring (NILM) addresses the issue of extracting device-level energy consumption information by monitoring the aggregated signal at one single measurement point without installing meters on each
individual device. Energy disaggregation can be formulated as a source separation problem where the aggregated signal is expressed as linear combination of basis vectors in a matrix factorization framework. In this work, we use machine learning in order to predict the pattern of energy consumption of each device during a day. 
This project is one part of our collaboration with [DOE](https://www.osti.gov/biblio/1265590-non-intrusive-load-monitoring-hvac-components-using-signal-unmixing).



###prerequisite:###
MATLAB R2015a

###Datasets###
(Temporay unavailable. Will be available soon after getting the required permissions. Sorry for the inconvenience!) 

We design two different experiments for evaluating our proposed algorithm. The first experiment is disaggregation of the whole home energy to the energy consumption of all the appliances at a residential home. 
For this part we use the current signal in [AMPds](http://ampds.org/): A Public Dataset for Load Disaggregation and Eco-Feedback Research for the first experiment. This dataset has most required features for performing an accurate disaggregation task.

The second experiment is based on designing a hierarchical scheme for disaggregating the whole building energy signal to the HVAC components signals (i.e., two compressors, two condenser fan and one indoor blower) in an industrial building (as shown below).

![Diag](blockdiag1.PNG)

## HVORUT dataset for NILM

For this experiment, we collected the data at Oak Ridge National Laboratory (ORNL) Flexible Research
Platform (FRP). FRP was constructed to enable research into building envelope materials, construction methods, building equipment, and building instrumentation, control, and fault detection.

Our experiment in this part consists of two major steps: 

1-Disaggregation of the power signal of whole building to power
signals of all the circuits and devices existing in the building.

There are 16 different devices (in FRP1), circuits and plugs in the building:
HVAC unit, 480/208 Transformer, lighting circuits: 1, 3, 5, 7, Plug circuits: 1, 3, 5, 7, cord reel circuit, lighting control box, exhaust fan, piping heat trace, exterior lighting (lighting and
emergency) and building control circuit. 

2-Decomposition of the obtained HVAC power signal from the previous step and estimating the power consumption profile of its components
including:
two compressors, two condenser fan and one indoor blower. The above Figure illustrates this hierarchical architecture. We collected the data in two different buildings (i.e., FRP1 and FRP2). The following is a brief  explanation about each.  

###FRP1 details###

In this part of the dataset the Bldg_Power(1) is the total power consumption of the building including
all the existing devices and plugs: 
Bldg_Power(2),	Bldg_Power(3),	Bldg_Power(4),	Bldg_Power(5),
Bldg_Power(6),	Bldg_Power(7),	Bldg_Power(8),	Bldg_Power(9),	Bldg_Power(10),	Bldg_Power(11),
Bldg_Power(12),	Bldg_Power(13),	Bldg_Power(14),	Bldg_Power(15),	Bldg_Power(16) and HVAC_Power(1).

The HVAC_Power(1) is the aggregated power signal for the HVAC unit.
HVAC_Power(2), HVAC_Power(3), HVAC_Power(4), HVAC_Power(5), HVAC_Power(6) and HVAC_Power(7) are the 
power signal of all the HVAC components (compressor, etc.). Sampling rate for this dataset is 2 samples
per minute. 

###FRP2 details###

In this data, W_BldgTot	is the total power signal (Watt) of the building including the power signals of 
W_Transf, W_Lights_dwn, W_EmLights_dwn, W_Lights_up, W_EmLights_up, W_WallHeater, W_Plugs102, W_Plugs103, W_Plugs104, W_Plugs, W_Plugs106, W_Plugs202, W_Plugs203, W_Plugs204,	W_Plugs205,	W_Plugs206,	W_PlugsExt,	W_DuctSmokeDet,	W_ExhaustFan, W_Unitary and W_RTU_Total.

The W_RTU_Total is the totall power signal of the RTU (HVAC). 

W_RTU_Comp1, W_RTU_Cond1, W_RTU_Comp2, W_RTU_Cond2 and W_RTU_EvapFan are the power signal of different 
components of HVAC unit. Sampling rate in this data is 2 samples per minute. 
	


Please see the [data folder](/data/?at=master) and ([the HVORUT dataset details.](/data/ORNL_data_info.zip)) for more details of all the 
collected data in different time spans during the year in FRP1 and FRP2. 
Please kindly cite our [paper](http://ieeexplore.ieee.org/abstract/document/7835299/) if you find this paper and dataset useful. 

###Demo###
Run the [`Demo.m`](/Demo.m) to see the result of disaggregation algorithm on the AMPds dataset. 
Please put the [`AMP_DATA.mat`](/AMP_DATA.mat) in the same folder when running the [`Demo.m`](/Demo.m).


* If parameter `training=1`, code performs the signal decomposition only without prediction. 

* For prediction: `training=0`.

* Please see the guide in the beginning of the demo code to see how you can apply different methods such as Non-negative Sparse coding and Elastic Net. 



###Results###
You should see the following results (and a lot more!) after running the demo. 

* Ground truth and estimated appliances signals using the S2K-NMF method for one random testing day (1440 minutes).

![f1](alldev2.png)

___
* The pie plots show that S2K-NMF achieves the best result for estimating the energy usage contribution of each device:

![f2](pie2.png)


___


* Ground truth (top figure) and estimated aggregated signal (middle
figure) and absolute difference between them (bottom figure) for the residential
home in one random testing day (1440 minutes) using the S2K-NMF
algorithm.
![f3](AGG_2.png)


___



###Useful links###

* [My NMF and Load disaggregation presentation](http://web.eecs.utk.edu/~arahimpo/NMF.pdf)
* [NILM Toolkit](http://nilmtk.github.io/)
* [NILM 2016 workshop](http://nilmworkshop.org/2016/)

###Citation:###

* [Non-Intrusive Energy Disaggregation Using Non-negative Matrix Factorization with Sum-to-k Constraint.](http://ieeexplore.ieee.org/abstract/document/7835299/)

IEEE Transactions on Power Systems
~~~~
@article{rahimpour2017non,
title={Non-Intrusive Energy Disaggregation Using Non-negative Matrix Factorization with Sum-to-k Constraint},
author={Rahimpour, Alireza and Qi, Hairong and Fugate, David and Kuruganti, Teja},
journal={IEEE Transactions on Power Systems},
year={2017},
publisher={IEEE}
} 
~~~~

* [Non-intrusive load monitoring of HVAC components using signal unmixing.](http://ieeexplore.ieee.org/abstract/document/7418350/)

IEEE Global Conference on Signal and Information Processing (GlobalSIP), 2015
~~~~
@inproceedings{rahimpour2015non,
title={Non-intrusive load monitoring of HVAC components using signal unmixing},
author={Rahimpour, Alireza and Qi, Hairong and Fugate, David and Kuruganti, Teja},
booktitle={IEEE Global Conference on Signal and Information Processing (GlobalSIP), 2015},
pages={1012--1016},
year={2015},
organization={IEEE}
}
~~~~

###Contact###

Please feel free to contact [Alireza Rahimpour](mailto:arahimpo@utk.edu) for more information about this project.
