namespace NeuralNetwork;

public abstract class FruitGeneration
{
        public static void GenerateRandomData(double[][] inputs, double[][] outputs, int numFruits)
        {
            // Generate random data for the specified number of fruits
            Random rand = new Random();
            int numPoisonous = (int)(numFruits * 0.5);
            int numNonPoisonous = (int)(numFruits * 0.5);

            for (int i = 0; i < numPoisonous; i++)
            {
                // Generate random poisonous fruit
                inputs[i] = GeneratePoisonousData(rand);

                var spikeLength = inputs[i][0];
                var spotSize = inputs[i][1];
                var color = inputs[i][2];
                var ripeness = inputs[i][3];

                outputs[i] = GenerateOutputData(spikeLength, spotSize, color, ripeness);
            }
            for (int i = numPoisonous; i < numPoisonous + numNonPoisonous; i++)
            {
                // Generate random non-poisonous fruit
                inputs[i] = GenerateNonPoisonousData(rand);

                var spikeLength = inputs[i][0];
                var spotSize = inputs[i][1];
                var color = inputs[i][2];
                var ripeness = inputs[i][3];

                outputs[i] = GenerateOutputData(spikeLength, spotSize, color, ripeness);
            }
        }
        
        public static double[] GeneratePoisonousData(Random rand)
        {
            // Generate random characteristics for each fruit
            double spikeLength = Math.Round(rand.NextDouble() * 0.49, 2) + 0.51; // Spike length between 0.51 and 1
            double spotSize = Math.Round(rand.NextDouble() * 0.29, 2); // Spot size between 0 and 0.29
            int color = rand.Next(100); // Color value between 0 and 99
            double ripeness = Math.Round(rand.NextDouble() * 0.64, 2); // Ripeness between 0 and 0.64

            // Store the characteristics in the input array
            var inputs = new double[] { spikeLength, spotSize, color, ripeness };
            return inputs;
        }

        public static double[] GenerateNonPoisonousData(Random rand)
        {
            // Generate random characteristics for each fruit
            double spikeLength = Math.Round(rand.NextDouble()*0.50, 2); // Spike length between 0 and 1
            double spotSize = Math.Round(rand.NextDouble()*0.7, 2)+0.3; // Spot size between 0 and 1
            int color = rand.Next(155)+100; // Color value between 0 and 255
            double ripeness = Math.Round(rand.NextDouble()*0.35, 2)*0.65; // Ripeness between 0 and 1

            // Store the characteristics in the input array
            var inputs = new double[] { spikeLength, spotSize, color, ripeness };
            return inputs;
        }

        private static double[] GenerateOutputData(double spikeLength, double spotSize, double color, double ripeness, bool? forcePoisonous = null)
        {
            // Determine if the fruit is poisonous based on spike length, spot size, color, and ripeness
            // For example, if spike length is more than 0.75, spot size is less than 0.3, color is dark (less than 100), and ripeness is high (more than 0.65), mark it as poisonous
            // Otherwise, mark it as non-poisonous
            var poisonousStatus = 0;
            {
                poisonousStatus = (spikeLength > 0.5 && spotSize < 0.3 && color < 100 && ripeness < 0.65) ? 1 : 0;
            }
            return new double[] { poisonousStatus };
        }
}