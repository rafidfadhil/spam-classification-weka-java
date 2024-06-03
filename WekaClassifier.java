import java.io.*;
import java.util.*;
import java.util.logging.Logger;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.meta.FilteredClassifier;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class WekaClassifier {

    private static Logger LOGGER = Logger.getLogger("WekaClassifier");

    private FilteredClassifier classifier;
    private Instances trainData;
    private ArrayList<Attribute> wekaAttributes;

    private static final String TRAIN_DATA = "dataset/train.txt";
    private static final String TRAIN_ARFF = "dataset/train.arff";
    private static final String TEST_DATA = "dataset/test.txt";
    private static final String TEST_ARFF = "dataset/test.arff";

    public WekaClassifier() {
        classifier = new FilteredClassifier();
        classifier.setClassifier(new NaiveBayesMultinomial());

        Attribute attributeText = new Attribute("text", (List<String>) null);
        ArrayList<String> classAttributeValues = new ArrayList<>();
        classAttributeValues.add("spam");
        classAttributeValues.add("ham");
        Attribute classAttribute = new Attribute("label", classAttributeValues);

        wekaAttributes = new ArrayList<>();
        wekaAttributes.add(classAttribute);
        wekaAttributes.add(attributeText);
    }

    public void transform() {
        try {
            trainData = loadRawDataset(TRAIN_DATA);
            saveArff(trainData, TRAIN_ARFF);

            StringToWordVector filter = new StringToWordVector();
            filter.setAttributeIndices("last");

            NGramTokenizer tokenizer = new NGramTokenizer();
            tokenizer.setNGramMinSize(1);
            tokenizer.setNGramMaxSize(1);
            tokenizer.setDelimiters("\\W");
            filter.setTokenizer(tokenizer);

            filter.setLowerCaseTokens(true);

            classifier.setFilter(filter);
        } catch (Exception e) {
            LOGGER.warning("Error transforming data: " + e.getMessage());
        }
    }

    public void fit() {
        try {
            classifier.buildClassifier(trainData);
        } catch (Exception e) {
            LOGGER.warning("Error fitting model: " + e.getMessage());
        }
    }

    public String predict(String text) {
        try {
            Instances newDataset = new Instances("predictiondata", wekaAttributes, 1);
            DenseInstance newinstance = new DenseInstance(2);
            newinstance.setDataset(newDataset);
            newDataset.setClassIndex(0); // Tetapkan indeks kelas untuk instance output
            newinstance.setValue(wekaAttributes.get(1), text);
            double pred = classifier.classifyInstance(newinstance);
            return newDataset.classAttribute().value((int) pred);
        } catch (Exception e) {
            LOGGER.warning("Error predicting: " + e.getMessage());
            return null;
        }
    }

    public String evaluate() {
        try {
            Instances testData;
            if (new File(TEST_ARFF).exists()) {
                testData = loadArff(TEST_ARFF);
            } else {
                testData = loadRawDataset(TEST_DATA);
                saveArff(testData, TEST_ARFF);
            }

            // Set class index
            testData.setClassIndex(0);

            Evaluation eval = new Evaluation(testData);
            eval.evaluateModel(classifier, testData);

            int spamIndex = testData.classAttribute().indexOfValue("spam");
            int hamIndex = testData.classAttribute().indexOfValue("ham");

            double[][] confusionMatrix = eval.confusionMatrix();
            StringBuilder sb = new StringBuilder();
            sb.append("\n=== Confusion Matrix ===\n\n");
            sb.append("\t").append("spam").append("\t").append("ham").append("\t").append("<-- classified as\n");
            sb.append("spam").append("\t").append((int) confusionMatrix[spamIndex][spamIndex]).append("\t").append((int) confusionMatrix[spamIndex][hamIndex]).append("\n");
            sb.append("ham").append("\t").append((int) confusionMatrix[hamIndex][spamIndex]).append("\t").append((int) confusionMatrix[hamIndex][hamIndex]).append("\n");

            return eval.toSummaryString() + sb.toString();
        } catch (Exception e) {
            LOGGER.warning("Error evaluating: " + e.getMessage());
            return null;
        }
    }

    public void loadModel(String fileName) {
        try {
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
            Object tmp = in.readObject();
            classifier = (FilteredClassifier) tmp;
            in.close();
            LOGGER.info("Loaded model: " + fileName);
        } catch (IOException | ClassNotFoundException e) {
            LOGGER.warning("Error loading model: " + e.getMessage());
        }
    }

    public void saveModel(String fileName) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
            out.writeObject(classifier);
            out.close();
            LOGGER.info("Saved model: " + fileName);
        } catch (IOException e) {
            LOGGER.warning("Error saving model: " + e.getMessage());
        }
    }

    public Instances loadRawDataset(String filename) {
        Instances dataset = new Instances("SMS spam", wekaAttributes, 10);
        dataset.setClassIndex(0);

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            for (String line; (line = br.readLine()) != null;) {
                String[] parts = line.split("\\s+", 2);
                if (!parts[0].isEmpty() && !parts[1].isEmpty()) {
                    DenseInstance row = new DenseInstance(2);
                    row.setValue(wekaAttributes.get(0), parts[0]);
                    row.setValue(wekaAttributes.get(1), parts[1]);
                    dataset.add(row);
                }
            }
        } catch (IOException e) {
            LOGGER.warning("Error loading raw dataset: " + e.getMessage());
        } catch (ArrayIndexOutOfBoundsException e) {
            LOGGER.info("Invalid row.");
        }
        return dataset;
    }

    public Instances loadArff(String fileName) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            ArffReader arff = new ArffReader(reader);
            Instances dataset = arff.getData();
            reader.close();
            return dataset;
        } catch (IOException e) {
            LOGGER.warning("Error loading ARFF: " + e.getMessage());
            return null;
        }
    }

    public void saveArff(Instances dataset, String filename) {
        try {
            ArffSaver arffSaverInstance = new ArffSaver();
            arffSaverInstance.setInstances(dataset);
            arffSaverInstance.setFile(new File(filename));
            arffSaverInstance.writeBatch();
        } catch (IOException e) {
            LOGGER.warning("Error saving ARFF: " + e.getMessage());
        }
    }
    public static void main(String[] args) throws Exception {
        final String MODEL = "models/sms.dat";

        WekaClassifier wt = new WekaClassifier();

        if (new File(MODEL).exists()) {
            wt.loadModel(MODEL);
        } else {
            wt.transform();
            wt.fit();
            wt.saveModel(MODEL);
        }

        // Add the following code to display the spam email message
        String emailSpam = "Ini adalah pesan email spam yang ingin Anda uji.";
        LOGGER.info("Pesan email: " + emailSpam);

        // Display prediction result
        String prediction = wt.predict(emailSpam);
        LOGGER.info("Hasil prediksi: " + (prediction.equals("spam") ? "SPAM" : "HAM"));

        // Run evaluation
        String evaluationResult = wt.evaluate();
        if (evaluationResult != null) {
            LOGGER.info("Evaluation Result:\n" + evaluationResult);
        }
    }
}
