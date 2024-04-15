namespace NeuralNetwork;

public class NeuralNetwork
{
    private static Random rand = new Random();
    // Weights for the connections between the input and hidden layers & hidden and output layers
    private double[,] weightsInputHidden = new double[4, 3];
    private double[,] weightsHiddenOutput = new double[3, 1];

    public NeuralNetwork()
    {
        // Initialize weights randomly
        for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            weightsInputHidden[i, j] = rand.NextDouble();
        
        for (int i = 0; i < 3; i++)
            weightsHiddenOutput[i, 0] = rand.NextDouble();
    }
    
    public double[] FeedForward(double[] inputs)
    {
        // Calculate the output of the hidden layer
        double[] hiddenLayerOutputs = new double[3];
        for (int i = 0; i < 3; i++) // For each neuron in the hidden layer
        for (int j = 0; j < 4; j++) // For each input
            hiddenLayerOutputs[i] += inputs[j] * weightsInputHidden[j, i];

        // Apply the activation function to the hidden layer outputs
        double output = 0;
        // Calculate the output of the output layer
        for (int i = 0; i < 3; i++)
            output += hiddenLayerOutputs[i] * weightsHiddenOutput[i, 0];
        // Apply the activation function to the output
        return new double[] { output };
    }

    public void Train(double[][] inputs, double[] outputs, int numEpochs)
    {
        // Train the neural network
        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            // For each training example
            for (int i = 0; i < inputs.Length; i++)
            {
                // Make a prediction
                double[] predictedOutputs = FeedForward(inputs[i]);
                double error = outputs[i] - predictedOutputs[0];

                // Adjust weights
                for (int j = 0; j < 4; j++)
                for (int k = 0; k < 3; k++)
                    weightsInputHidden[j, k] += inputs[i][j] * error * 0.01;
                
                for (int j = 0; j < 3; j++)
                    weightsHiddenOutput[j, 0] += error * 0.01;
            }
        }
    }
}