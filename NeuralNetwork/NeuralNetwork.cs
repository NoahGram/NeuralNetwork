namespace NeuralNetwork;

public class NeuralNetwork
{
    public enum ActivationFunction
    {
        Sigmoid,
        HyperbolicTangent,
        ReLu
    }
    
    private static Random _rand = new Random();
    // Weights between input and hidden layers
    private double[,] _weightsInputHidden;
    // Weights between hidden and output layers
    private double[,] _weightsHiddenOutput;
    // Placeholder for the activation function
    private ActivationFunction _activationFunction;

    public NeuralNetwork(int inputNodes, int hiddenNodes, ActivationFunction activationFunction)
    {
        this._activationFunction = activationFunction;
        // Initialize weights
        _weightsInputHidden = new double[inputNodes, hiddenNodes];
        _weightsHiddenOutput = new double[hiddenNodes, 1];

        // Initialize weights randomly
        for (int i = 0; i < inputNodes; i++)
            for (int j = 0; j < hiddenNodes; j++)
                _weightsInputHidden[i, j] = _rand.NextDouble();

        for (int i = 0; i < hiddenNodes; i++)
            _weightsHiddenOutput[i, 0] = _rand.NextDouble();
    }
    
    private double Sigmoid(double x)
    {
        // Maps any value to a value between 0 and 1
        return 1 / (1 + Math.Exp(-x));
    }
    
    private double HyperbolicTangent(double x)
    {
        // Maps any value to a value between -1 and 1
        return Math.Tanh(x);
    }

    private double ReLu(double x)
    {
        // Maps any negative value to 0 and keeps positive values as they are
        return Math.Max(0, x);
    }
    
    private double ApplyActivationFunction(double x)
    {
        switch (_activationFunction)
        {
            case ActivationFunction.Sigmoid:
                return Sigmoid(x);
            case ActivationFunction.HyperbolicTangent:
                return HyperbolicTangent(x);
            case ActivationFunction.ReLu:
                return ReLu(x);
            default:
                throw new ArgumentException("Invalid activation function");
        }
    }
    
    public double[] FeedForward(double[] inputs)
    {
        // Calculate the outputs of the hidden layer
        double[] hiddenLayerOutputs = new double[_weightsInputHidden.GetLength(1)];
        for (int i = 0; i < _weightsInputHidden.GetLength(1); i++)
            for (int j = 0; j < _weightsInputHidden.GetLength(0); j++)
                hiddenLayerOutputs[i] += inputs[j] * _weightsInputHidden[j, i];

        // Apply the activation function to the hidden layer outputs
        // Get the output of Hidden Layer
        
        // Apply the activation function to the hidden layer outputs
        for (int i = 0; i < hiddenLayerOutputs.Length; i++)
            hiddenLayerOutputs[i] = ApplyActivationFunction(hiddenLayerOutputs[i]);

        // Calculate the output of the neural network
        double output = 0;
        for (int i = 0; i < _weightsHiddenOutput.GetLength(0); i++)
            output += hiddenLayerOutputs[i] * _weightsHiddenOutput[i, 0];

        // Apply the activation function to the output
        output = ApplyActivationFunction(output);

        return new double[] { output };
    }

    public void Train(double[][] inputs, double[][] outputs, int numIterations, double lambda)
    {
        // Simple random search algorithm to train the network
        for (int iteration = 0; iteration < numIterations; iteration++)
        {
            // Logger per 100 iterations
            Console.WriteLine($"Iteration {iteration + 1}/{numIterations}");
            // Randomly adjust the weights
            for (int i = 0; i < _weightsInputHidden.GetLength(0); i++)
            for (int j = 0; j < _weightsInputHidden.GetLength(1); j++)
            {
                // Save the old weight
                double oldWeight = _weightsInputHidden[i, j];

                // Calculate the old error
                double oldError = CalculateMeanSquaredError(inputs, outputs);

                // Randomly adjust the weight and apply regularization
                _weightsInputHidden[i, j] += (_rand.NextDouble() - 0.5) * 0.01 - lambda * _weightsInputHidden[i, j];

                // Calculate the new error
                double newError = CalculateMeanSquaredError(inputs, outputs);

                // If the new weights do not improve the performance, revert to the old weight
                if (newError > oldError)
                    _weightsInputHidden[i, j] = oldWeight;
            }

            for (int i = 0; i < _weightsHiddenOutput.GetLength(0); i++)
            {
                // Save the old weight
                double oldWeight = _weightsHiddenOutput[i, 0];

                // Calculate the old error
                double oldError = CalculateMeanSquaredError(inputs, outputs);

                // Randomly adjust the weight and apply regularization
                _weightsHiddenOutput[i, 0] += (_rand.NextDouble() - 0.5) * 0.01 - lambda * _weightsHiddenOutput[i, 0];

                // Calculate the new error
                double newError = CalculateMeanSquaredError(inputs, outputs);

                // If the new weights do not improve the performance, revert to the old weight
                if (newError > oldError)
                    _weightsHiddenOutput[i, 0] = oldWeight;
            }
        }
    }

    private double CalculateMeanSquaredError(double[][] inputs, double[][] outputs)
    {
        // Calculate the mean squared error
        double totalError = 0;

        // For each training example
        for (int i = 0; i < inputs.Length; i++)
        {
            // Make a prediction
            double[] predictedOutputs = FeedForward(inputs[i]);

            // For each output value
            for (int j = 0; j < outputs[i].Length; j++)
            {
                double error = outputs[i][j] - predictedOutputs[j];

                // Add the squared error to the total error
                totalError += Math.Pow(error, 2);
            }
        }

        // Return the mean squared error
        return totalError / (inputs.Length * outputs[0].Length);
    }
    
    public string GetActivationFunction()
    {
        return _activationFunction.ToString();
    }
}