#Inverse problems
##Precipitate dissolution model of an aluminium alloy

```c++
#include <iostream>
#include <time.h>

#include "../../source/opennn.h"

#include "precipitate_dissolution.h"

using namespace OpenNN;

int main(void)
{
    PrecipitateDissolution precipitate_dissolution;

    precipitate_dissolution.set_reference_temperature(573.16); // K
    precipitate_dissolution.set_reference_time(1000); // s

    precipitate_dissolution.set_minimum_Vickers_hardness(66.9124);
    precipitate_dissolution.set_maximum_Vickers_hardness(151.9350);

    DataSet data_set;

    data_set.load_data("../data/precipitate_dissolution/DataAA-2014-T6.dat");

    VariablesInformation* variables_information_pointer = data_set.get_variables_information_pointer();

    variables_information_pointer->set(2, 1);

    variables_information_pointer->set_name(0, "temperature");
    variables_information_pointer->set_name(1, "time");
    variables_information_pointer->set_name(2, "vickers_hardness");

    variables_information_pointer->set_units(0, "celsius");
    variables_information_pointer->set_units(1, "s");
    variables_information_pointer->set_units(2, "none");

    const Matrix<double> input_data = data_set.arrange_input_data();

    NeuralNetwork neural_network(1, 3, 1);

    neural_network.construct_inputs_outputs_information();

    InputsOutputsInformation* inputs_outputs_information_pointer = neural_network.get_inputs_outputs_information_pointer();

    inputs_outputs_information_pointer->set_input_name(0, "log(t/t*)");
    inputs_outputs_information_pointer->set_output_name(0, "1-f/f0");

    inputs_outputs_information_pointer->set_input_units(0, "none");
    inputs_outputs_information_pointer->set_output_units(0, "none");

    MultilayerPerceptron* multilayer_perceptron_pointer = neural_network.get_multilayer_perceptron_pointer();

    multilayer_perceptron_pointer->initialize_parameters_normal(0.0, 1.0e-2);

    neural_network.construct_scaling_layer();

    ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

    scaling_layer_pointer->set_minimum(0, -6.0);
    scaling_layer_pointer->set_maximum(0, 6.0);

    neural_network.construct_unscaling_layer();

    UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();

    unscaling_layer_pointer->set_minimum(0, 0.0);
    unscaling_layer_pointer->set_maximum(0, 1.0);

    neural_network.construct_bounding_layer();

    BoundingLayer* bounding_layer_pointer = neural_network.get_bounding_layer_pointer();

    bounding_layer_pointer->set_lower_bound(0, 0.0);
    bounding_layer_pointer->set_upper_bound(0, 1.0);

    neural_network.construct_independent_parameters();

    IndependentParameters* independent_parameters_pointer = neural_network.get_independent_parameters_pointer();

    independent_parameters_pointer->set_parameters_number(1);
    independent_parameters_pointer->set_name(0, "effective_activation_energy");

    independent_parameters_pointer->set_minimum(0, 100.0);
    independent_parameters_pointer->set_maximum(0, 200.0);

    independent_parameters_pointer->set_lower_bound(0, 0.0);

    independent_parameters_pointer->set_parameter(0, 150.0);

    precipitate_dissolution.calculate_dependent_variables(neural_network, input_data);

    PerformanceFunctional performance_functional(&neural_network);

    performance_functional.construct_objective_term(PerformanceFunctional::INVERSE_SUM_SQUARED_ERROR, &precipitate_dissolution, &data_set);

    std::cout << performance_functional.calculate_gradient() << std::endl;

    TrainingStrategy training_strategy(&performance_functional);

    neural_network.save("../data/AA-7449-T79/NeuralNetworkAA-7449-T79.dat");

    return(0);
}
```

![Alt text](images/PrecipitateDissolutionModel.gif)

[Download data set](data/DataAA-2014-T6.dat)

Mathematical model
<br>
<br>
<br>
Data set
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
Neural network
<br>
<br>
<br>
Inputs-outputs information
<br>
<br>
<br>
<br>
<br>
<br>
<br>
Multilayer perceptron
<br>
<br>
<br>
Scaling layer
<br>
<br>
<br>
<br>
<br>
<br>
<br>
Unscaling layer
<br>
<br>
<br>
<br>
<br>
Bounding layer
<br>
<br>
<br>
<br>
<br>
Independent parameters
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
Performance functional

Objective term
<br>
<br>
<br>
Training strategy

Save results
