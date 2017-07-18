# Non-Intrusive Load Monitoring #
![LD](/LD.png)

Energy disaggregation or Non-Intrusive Load Monitoring (NILM) addresses the issue of extracting device-level energy consumption information by monitoring the aggregated signal at one single measurement point without installing meters on each individual device. Energy disaggregation can be formulated as a source separation problem where the aggregated signal is expressed as linear combination of basis vectors in a matrix factorization framework.

Paper: 
[Non-Intrusive Energy Disaggregation Using Non-negative Matrix Factorization with Sum-to-k Constraint.](http://ieeexplore.ieee.org/abstract/document/7835299/)


###prerequisite:###
MATLAB R2015a

###Datasets###
We design two different experiments for
evaluating our proposed algorithm. The first experiment is
disaggregation of the whole home energy to the energy consumption
of all the appliances at a residential home. 
For this part we use the [AMPds](http://ampds.org/): A Public Dataset for
Load Disaggregation and Eco-Feedback Research for
the first experiment. This dataset has most required features
for performing an accurate disaggregation task.

The second experiment is designing a hierarchical scheme for
disaggregating the whole building energy signal to the HVAC
components signals in an industrial building.

![Diag](blockdiag1.PNG)

For this experiment, the data was collected on the
Oak Ridge National Laboratory (ORNL) Flexible Research
Platform (FRP1). FRP1 was constructed to enable
research into building envelope materials, construction methods,
building equipment, and building instrumentation, control,
and fault detection. Please see [(The ORNL dataset details.)](/data/ORNL_data_info.zip) for details of all the 
collected data in different time spans during the year in FRP1 and FRP2. 

###Demo###
Run the [`Demo.m`](/Demo.m) to see the result of disaggregation algorithm on the AMPds dataset. 
Please put the [`AMP_DATA.mat`](/AMP_DATA.mat) in the same folder when running the [`Demo.m`](/Demo.m).


* If parameter `training=1`, code performs the signal decomposition only without prediction. 

* For prediction: `training=0`.

* See the guide in the begining of the demo code to see how you can apply different methods such as Non-negative Sparse coding and Elastic Net. 



###Results###

###Useful links###

###Citation:###
###Contact###
```
#
def wiki_rocks(text): formatter = lambda t: "funky"+t return formatter(text) 

```

~~~~
This is a code block, fenced-style
~~~~

~~text that has been struckthrough~~
