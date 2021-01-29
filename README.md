# Artificial Neural Network for Multiclass Classification
This is a simple 2-layer ANN made from scratch in Python. It accepts any number of numerical inputs and outputs a classification, from any sized class set.

## Demo
By using the ``-v`` or ``--visualise`` argument, a visualisation of the network is rendered using Pygame.<br/>
The application runs considerably slower when displaying the visualisation.

![Visual Demonstration](https://github.com/PDorrian/classification-neural-network/blob/master/demo.gif)

## Installing and Using the Program
To install this application, simply clone the repository and install Pygame using the command line.
```pip install pygame```
Once complete, simply run the application with the following command:
```python application.py -i <input file> -a <classification attribute>```

### Arguments
#### Required Arguments
| Argument | Description |
|--------------|-----------------------------------------------|
| -i, --input | Input file, CSV format required with headings. |
| -a, --attribute | Heading of column used for classification. |

#### Optional Arguments
| Argument | Description | Default Value |
|----------|-------------|:-------------:|
| -lr, --learningrate | Set the learning rate for backpropagation. | 5 |
| -iter, --iterations | Set the number of iterations the network trains for. | 5000 |
| -hn, --hiddennodes | Set the number of nodes in the hidden layer. | 5 |
| -r, --repeat | Set the number of times to repeat. | 1 |
| -o, --output | Set the location of the output file. | Main directory |
| -v, --visualise | Visualise the network while it is trained. | False |