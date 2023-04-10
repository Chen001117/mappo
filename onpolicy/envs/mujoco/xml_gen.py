def get_xml(dog_num=1, obs_num=1, anchor_id=None):
    assert len(anchor_id) == dog_num, "length of anchor id should be equal to the dog number"
    strings = \
    """
<mujoco model="navigation">
  
  <size njmax="3000" nconmax="1500"/>
  <option integrator="RK4" timestep="0.01" collision="predefined"/>

  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="5.2 5.2 5" type="plane" material="MatPlane"/>
    
    <body name="load" pos="0 0 0.35">
      <site name="load" pos="0 0 0"/>
      <camera name="camera" mode="trackcom" pos="0 0. 10." xyaxes="1 0 0 0 1 0"/>
      <joint axis="1 0 0" limited="false" name="load_axisx" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" limited="false" name="load_axisy" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" limited="false" name="load_rootz" pos="0 0 0" type="hinge"/>
      <joint axis="0 0 1" limited="false" name="load_axisz" pos="0 0 0" type="slide"/>
      <geom mass="0.1" size="0.3 0.3 0.3" name="load" type="box" rgba="0.55 0.27 0.07 1." friction="1 0.005 0.001" />
    """
    for i in range(dog_num):
       if anchor_id[i] == 0:
         x_coor, y_coor = 0.3, 0.
       elif anchor_id[i] == 1:
         x_coor, y_coor = 0., 0.3
       elif anchor_id[i] == 2:
         x_coor, y_coor = -0.3, 0.
       elif anchor_id[i] == 3:
         x_coor, y_coor = 0., -0.3
       strings += \
    """
      <site name="load_dog_{:02d}" pos="{} {} 0"/>
    """.format(i, x_coor, y_coor)
    
    strings += \
    """
    </body>
    """

    for i in range(dog_num):
      strings += \
    """
    <body name="dog{:02d}" pos="0 0 0.2">
      <site name="dog{:02d}" pos="0 0 0"/>
      <joint axis="1 0 0" limited="false" name="dog{:02d}_axisx" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" limited="false" name="dog{:02d}_axisy" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" limited="false" name="dog{:02d}_rootz" pos="0 0 0" type="hinge"/>
      <joint axis="0 0 1" limited="false" name="dog{:02d}_axisz" pos="0 0 0" type="slide"/>
      <geom mass="13." size="0.325 0.15 0.15" name="dog{:02d}" type="box" rgba="0.8 0.4 0. 1" friction="1 0.005 0.001" />
    </body>
    """.format(i,i,i,i,i,i,i)
    
    for i in range(obs_num):
      strings += \
    """
    <body name="obstacle{:02d}" pos="0 0 0">
      <joint axis="1 0 0" limited="false" name="obs{:02d}_axisx" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" limited="false" name="obs{:02d}_axisy" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" limited="false" name="obs{:02d}_rootz" pos="0 0 0" type="hinge"/>
      <joint axis="0 0 1" limited="false" name="obs{:02d}_axisz" pos="0 0 0" type="slide"/>
      <geom mass="1000" size="0.5 0.5 0.5" name="obstacle{:02d}" type="box" rgba="0. 0. 1. 1" friction="1 0.005 0.0001" />
    </body>
    """.format(i,i,i,i,i,i)
      
    
    strings += \
    """
    <body name="wall0" pos="0 5.1 0.5">
      <joint axis="1 0 0" limited="false" name="wall0_axisx" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" limited="false" name="wall0_axisy" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" limited="false" name="wall0_axisz" pos="0 0 0" type="slide"/>
      <geom mass="1000" size="5. 0.2 0.5" name="wall0" type="box" rgba="0. 0. 1. 1" friction="1 0.005 0.0001" />
    </body>
    <body name="wall1" pos="0 -5.1 0.5">
      <joint axis="1 0 0" limited="false" name="wall1_axisx" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" limited="false" name="wall1_axisy" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" limited="false" name="wall1_axisz" pos="0 0 0" type="slide"/>
      <geom mass="1000" size="5. 0.2 0.5" name="wall1" type="box" rgba="0. 0. 1. 1" friction="1 0.005 0.0001" />
    </body>
    <body name="wall2" pos="5.1 0 0.5">
      <joint axis="1 0 0" limited="false" name="wall2_axisx" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" limited="false" name="wall2_axisy" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" limited="false" name="wall2_axisz" pos="0 0 0" type="slide"/>
      <geom mass="1000" size="0.2 5. 0.5" name="wall2" type="box" rgba="0. 0. 1. 1" friction="1 0.005 0.0001" />
    </body>
    <body name="wall3" pos="-5.1 0 0.5">
      <joint axis="1 0 0" limited="false" name="wall3_axisx" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" limited="false" name="wall3_axisy" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" limited="false" name="wall3_axisz" pos="0 0 0" type="slide"/>
      <geom mass="1000" size="0.2 5. 0.5" name="wall3" type="box" rgba="0. 0. 1. 1" friction="1 0.005 0.0001" />
    </body>
    """

    strings += \
    """
    <body name="destination" pos="0 0 0">
      <site name="destination" pos="0 0 0"/>
      <joint axis="1 0 0" limited="false" name="destinationx" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" limited="false" name="destinationy" pos="0 0 0" type="slide"/>
      <geom name="destination" pos="0 0 0" size="0.01 0.01 0.01" type="box"/>
    </body>

  </worldbody>

  <tendon>
    <spatial width="0.03" rgba=".95 .0 .0 1" limited="true" range="0 100."> 
      <site site="load"/>
      <site site="destination"/>
    </spatial>
  """
    for i in range(dog_num):
      strings += \
  """
    <spatial width="0.03" rgba=".95 .95 .95 1" limited="true" range="0 1."> 
      <site site="load_dog_{:02d}"/>
      <site site="dog{:02d}"/>
    </spatial>
  """.format(i, i)
    strings += \
    """
  </tendon>
    """
    for i in range(dog_num):
       strings += \
    """
  <actuator>
    <motor ctrllimited="true" ctrlrange="-500.0 500.0" joint=dog{:02d}_axisx/>
    <motor ctrllimited="true" ctrlrange="-500.0 500.0" joint="dog{:02d}_axisy"/>
    <motor ctrllimited="true" ctrlrange="-100.0 100.0" joint="dog{:02d}_rootz"/>
  </actuator>
    """.format(i,i,i)
  
    strings += \
    """
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0" width="100" height="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="5 5" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>

  <contact>
  <pair geom1="load" geom2="floor"/>
  <pair geom1="load" geom2="wall0"/>
  <pair geom1="load" geom2="wall1"/>
  <pair geom1="load" geom2="wall2"/>
  <pair geom1="load" geom2="wall3"/>
  <pair geom1="wall0" geom2="floor"/>
  <pair geom1="wall1" geom2="floor"/>
  <pair geom1="wall2" geom2="floor"/>
  <pair geom1="wall3" geom2="floor"/>
  """
    
    for i in range(obs_num):
      for j in range(dog_num):
        strings += \
  """
  <pair geom1="dog{:02d}" geom2="obstacle{:02d}"/>
  """.format(j,i)
        
    for i in range(obs_num):
      strings += \
  """
  <pair geom1="obstacle{:02d}" geom2="floor"/>
  <pair geom1="obstacle{:02d}" geom2="load"/>
  """.format(i,i)
      
    for i in range(dog_num):
      for j in range(dog_num):
        if i != j:
          strings += \
  """
  <pair geom1="dog{:02d}" geom2="dog{:02d}"/>
  """.format(i,j)
        
      strings += \
  """
  <pair geom1="dog{:02d}" geom2="floor"/>
  <pair geom1="dog{:02d}" geom2="load"/>
  <pair geom1="dog{:02d}" geom2="wall0"/>
  <pair geom1="dog{:02d}" geom2="wall1"/>
  <pair geom1="dog{:02d}" geom2="wall2"/>
  <pair geom1="dog{:02d}" geom2="wall3"/>
  """.format(i,i,i,i,i,i)
      
    strings += \
  """
  </contact>
  </mujoco>
  """
    return strings