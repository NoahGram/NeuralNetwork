namespace NeuralNetwork;

public class NeuralNetwork
{
    public enum ActivationFunction
    {
        Sigmoid,
        HyperbolicTangent,
        ReLU
    }
    
    private static Random rand = new Random();
    // Weights between input and hidden layers
    private double[,] weightsInputHidden;
    // Weights between hidden and output layers
    private double[,] weightsHiddenOutput;
    // Placeholder for the activation function
    private ActivationFunction activationFunction;

    public NeuralNetwork(int inputNodes, int hiddenNodes, ActivationFunction activationFunction)
    {
        this.activationFunction = activationFunction;
        // Initialize weights
        weightsInputHidden = new double[inputNodes, hiddenNodes];
        weightsHiddenOutput = new double[hiddenNodes, 1];

        // Initialize weights randomly
        for (int i = 0; i < inputNodes; i++)
        for (int j = 0; j < hiddenNodes; j++)
            weightsInputHidden[i, j] = rand.NextDouble();

        for (int i = 0; i < hiddenNodes; i++)
            weightsHiddenOutput[i, 0] = rand.NextDouble();
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

    private double ReLU(double x)
    {
        // Maps any negative value to 0 and keeps positive values as they are
        return Math.Max(0, x);
    }
    
    private double ApplyActivationFunction(double x)
    {
        switch (activationFunction)
        {
            case ActivationFunction.Sigmoid:
                return Sigmoid(x);
            case ActivationFunction.HyperbolicTangent:
                return HyperbolicTangent(x);
            case ActivationFunction.ReLU:
                return ReLU(x);
            default:
                throw new ArgumentException("Invalid activation function");
        }
    }
    
    public double[] FeedForward(double[] inputs)
    {
        // Calculate the outputs of the hidden layer
        double[] hiddenLayerOutputs = new double[weightsInputHidden.GetLength(1)];
        for (int i = 0; i < weightsInputHidden.GetLength(1); i++)
        for (int j = 0; j < weightsInputHidden.GetLength(0); j++)
            hiddenLayerOutputs[i] += inputs[j] * weightsInputHidden[j, i];

        // Apply the activation function to the hidden layer outputs
        // Get the output of Hidden Layer
        
        // Apply the activation function to the hidden layer outputs
        for (int i = 0; i < hiddenLayerOutputs.Length; i++)
            hiddenLayerOutputs[i] = ApplyActivationFunction(hiddenLayerOutputs[i]);

        // Calculate the output of the neural network
        double output = 0;
        for (int i = 0; i < weightsHiddenOutput.GetLength(0); i++)
            output += hiddenLayerOutputs[i] * weightsHiddenOutput[i, 0];

        // Apply the activation function to the output
        output = ApplyActivationFunction(output);

        return new double[] { output };
    }

    public void Train(double[][] inputs, double[][] outputs, int numIterations)
    {
        // Simple random search algorithm to train the network
        for (int iteration = 0; iteration < numIterations; iteration++)
        {
            // Randomly adjust the weights
            for (int i = 0; i < weightsInputHidden.GetLength(0); i++)
            for (int j = 0; j < weightsInputHidden.GetLength(1); j++)
            {
                // Save the old weight
                double oldWeight = weightsInputHidden[i, j];

                weightsInputHidden[i, j] += (rand.NextDouble() - 0.5) * 0.01;
                // If the new weights do not improve the performance, revert to the old weight
                if (CalculateMeanSquaredError(inputs, outputs) < CalculateMeanSquaredError(inputs, outputs))
                    weightsInputHidden[i, j] = oldWeight;
            }
            
            for (int i = 0; i < weightsHiddenOutput.GetLength(0); i++)
            {
                // Save the old weight
                double oldWeight = weightsHiddenOutput[i, 0];

                // Randomly adjust the weight
                weightsHiddenOutput[i, 0] += (rand.NextDouble() - 0.5) * 0.01;

                // If the new weights do not improve the performance, revert to the old weight
                if (CalculateMeanSquaredError(inputs, outputs) < CalculateMeanSquaredError(inputs, outputs))
                    weightsHiddenOutput[i, 0] = oldWeight;
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
        return activationFunction.ToString();
    }
}