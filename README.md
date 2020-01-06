# Policy Gradients Temperature Control
![temperature control](https://github.com/smellslikeml/rl_thermo/blob/master/assets/pg_temp.png?raw=true)

This repo includes resources to implement an agent that learns a policy function in an online manner to specialize temperature control with policy gradients. This method is flexible enough to use various types of appliances (simple heaters, fans, AC units) to keep temperature in a space at a given set point.

## Getting Started
The ```config.ini``` will contain all of the default variables needed for training and data logging. Update the PATHS section with the appropriate paths if you are modifying from the defaults.

### Prerequisites
* [Tensorflow](https://www.tensorflow.org)
* [Sqlite3](https://docs.python.org/2/library/sqlite3.html)
* [ConfigParser](https://docs.python.org/3/library/configparser.html)
* [Adafruit BME280](https://www.adafruit.com/product/2652)

### Installing
This example assumes you are using a BME280 temperature sensor connected via GPIO. You can use any other source of temperature and modify the main.py script accordingly. Pip install the python dependencies:
```bash
# Assuming python 3.x as default

$ pip install -r requirements
```

## Run

Simply run the ```main.py``` script to learn a policy function in an online fashion.
``` bash
$ python3 main.py
```
The ```utils/``` directory contains resources to switch a TP-Link smartplug on/off given the output action by the agent.

## Deployment

This minimal implementation can run on embedded devices as small as a Raspberry Pi Zero W. 

## Contributing

Please read [CONTRIBUTING](CONTRIBUTING) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details

## References

* [Policy Gradients Temperature Control in use](https://www.hackster.io/kindbot/kindbot-home-garden-automation-hub-4c218a#toc-kindbot-cool-2)
    * [Code](https://github.com/smellslikeml/kindbot)
* [Ch.16 Reinforcement Learning: Hands-on Machine Learning with Scikit-Learn and Tensorflow](https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb)
* [pyHS100](https://github.com/GadgetReactor/pyHS100)
