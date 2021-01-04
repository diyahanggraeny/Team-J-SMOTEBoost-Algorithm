
package weka.classifiers.meta;


import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.Sourcable;
import weka.core.Capabilities;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;



public class SMOTEBoost extends RandomizableIteratedSingleClassifierEnhancer
		implements WeightedInstancesHandler, TechnicalInformationHandler {

	
	private static int maksimum_iterasi = 10;
	protected double[] bobot_votes;
	protected int iterasi_sukses;
	protected int bobot_training = 100;
	protected int jumlah_class;
	protected String SMOTE_indeks = "0";
	protected int SMOTE_tetangga = 5;
	protected double SMOTE_persen = 100.0;
	protected int SMOTE_RandomSeed = 1;
	private static final long versi_serial = -1397262307604906824L;

	
	public SMOTEBoost() {

		m_Classifier = new weka.classifiers.trees.DecisionStump();
	}


	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.INPROCEEDINGS);
		result.setValue(Field.AUTHOR,
				"Nitesh V. Chawla, Aleksandar Lazarevic, Lawrence O.Hall and Kevin W. Bowyer");
		result.setValue(Field.TITLE,
				"SMOTEBoost: Improving Prediction of the Minority Class in Boosting");
		result.setValue(
				Field.BOOKTITLE,
				"7th European Conference on Principles and Practice of Knowledge Discovery in Databases (PKDD)");
		result.setValue(Field.YEAR, "2003");
		result.setValue(Field.PAGES, "107-119");
		result.setValue(Field.ADDRESS, "Dubrovnik, Croatia");

		return result;
	}

	
	protected String defaultClassifierString() {

		return "weka.classifiers.trees.DecisionStump";
	}

	/*
	 * data = input instances
	 * kuantil = nilai kuantil yang dipilih
	 * return instances yang dipilih
	 */

	protected Instances bobotkuantil(Instances data, double kuantil) {

		int indeksnum = data.numInstances();
		Instances datatraining = new Instances(data, indeksnum);
		double[] weights = new double[indeksnum];

		double jumlahweights = 0;
		for (int i = 0; i < indeksnum; i++) {
			weights[i] = data.instance(i).weight();
			jumlahweights += weights[i];
		}
		double hasil_weights = jumlahweights * kuantil;
		int[] indeks_urut = Utils.sort(weights);

		// memilih instances
		jumlahweights = 0;
		for (int i = indeksnum - 1; i >= 0; i--) {
			Instance instance = (Instance) data.instance(indeks_urut[i])
					.copy();
			datatraining.add(instance);
			jumlahweights += weights[indeks_urut[i]];
			if ((jumlahweights > hasil_weights)
					&& (i > 0)
					&& (weights[indeks_urut[i]] != weights[indeks_urut[i - 1]])) {
				break;
			}
		}
		if (data_error) {
			System.err.println("Pilih " + datatraining.numInstances()
					+ " dari " + indeksnum);
		}
		return datatraining;
	}

	/**
	 * Return elemen pilihan dari option
	 */
	public Enumeration new_vector() {

		Vector vektor_baru = new Vector();

		vektor_baru.addElement(new Option(
				"\tPersentasi bobot untuk training.\n"
						+ "(Kurangi sekitar 90 dari 100)", "P",
				1, "-P <num>"));

		Enumeration enu = super.new_vector();
		while (enu.hasMoreElements()) {
			vektor_baru.addElement(enu.nextElement());
		}

		return vektor_baru.elements();
	}

	
	public void setOptions(String[] options) throws Exception {

		String pilihan = Utils.getOption('P', options);
		if (pilihan.length() != 0) {
			setBobot_Awal(Integer.parseInt(pilihan));
		} else {
			setBobot_Awal(100);
		}

		String strseed = Utils.getOption("smoteS", options);
		if (strseed.length() != 0) {
			setSMOTE_RandomSeed(Integer.parseInt(strseed));
		}

		String strpersen = Utils.getOption("smoteP", options);
		if (strpersen.length() != 0) {
			setSMOTE_Persentasi(new Double(strpersen).doubleValue());
		}

		String strtetangga = Utils.getOption('K', options);
		if (strtetangga.length() != 0) {
			setSMOTE_Tetangga(Integer.parseInt(strtetangga));
		}

		String strindeks = Utils.getOption('C', options);
		if (strindeks.length() != 0) {
			setSMOTE_Indeks(strindeks);
		}

		super.setOptions(options);
	}

	
	public String[] getOptions() {
		Vector result;
		String[] options;
		int i;

		result = new Vector();

		result.add("-C");
		result.add(getSMOTE_Indeks());

		result.add("-K");
		result.add("" + getSMOTE_Tetangga());

		result.add("-smoteP");
		result.add("" + getSMOTE_Persentasi());

		result.add("-smoteS");
		result.add("" + getSMOTE_RandomSeed());

		result.add("-P");
		result.add("" + getBobot_Awal());

		options = super.getOptions();
		for (i = 0; i < options.length; i++)
			result.add(options[i]);

		return (String[]) result.toArray(new String[result.size()]);
	}

	public void setBobot_Awal(int threshold) {

		bobot_training = threshold;
	}


	public int getBobot_Awal() {

		return bobot_training;
	}

	public int getSMOTE_RandomSeed() {
		return SMOTE_RandomSeed;
	}

	public void setSMOTE_RandomSeed(int value) {
		SMOTE_RandomSeed = value;
	}

	public void setSMOTE_Persentasi(double value) {
		if (value >= 0)
			SMOTE_persen = value;
		else
			System.err.println("Persentasi harus bernilai >= 0!");
	}

	public double getSMOTE_Persentasi() {
		return SMOTE_persen;
	}


	public void setSMOTE_Tetangga(int value) {
		if (value >= 1)
			SMOTE_tetangga = value;
		else
			System.err.println("Pilih minimal satu tetangga!");
	}

	public int getSMOTE_Tetangga() {
		return SMOTE_tetangga;
	}

	public void setSMOTE_Indeks(String value) {
		SMOTE_indeks = value;
	}

	public String getSMOTE_Indeks() {
		return SMOTE_indeks;
	}

	public SMOTE initSMOTE() {

		SMOTE smote = new SMOTE();

		smote.setRandomSeed(this.getSMOTE_RandomSeed());
		smote.setPercentage(this.getSMOTE_Persentasi());
		smote.setNearestNeighbors(this.getSMOTE_Tetangga());
		smote.setClassValue(this.getSMOTE_Indeks());

		return smote;
	}



	//* Boosting method.*//

	public void setClassifier(Instances data) throws Exception {

		super.setClassifier(data);

		getCapabilities().testWithFail(data);

		data = new Instances(data);
		data.deleteWithMissingClass();

		if (data.numAttributes() == 1) {
			System.err
					.println("Gagal membangun model!");
			return;
		} 

		jumlah_class = data.numClasses();
		if ((m_Classifier instanceof WeightedInstancesHandler)) {
			SMOTEBoostAlgorithm(data);
		} 
	}



	protected void setBobot(Instances training, double Bobot)
			throws Exception {

		double BobotLama, BobotBaru;

		BobotLama = training.sumOfWeights();
		Enumeration enu = training.enumerateInstances();
		while (enu.hasMoreElements()) {
			Instance instance = (Instance) enu.nextElement();
			if (!Utils.eq(m_Classifier[iterasi_sukses]
					.classifyInstance(instance), instance.classValue()))
				instance.setWeight(instance.weight() * Bobot);
		}

		BobotBaru = training.sumOfWeights();
		enu = training.enumerateInstances();
		while (enu.hasMoreElements()) {
			Instance instance = (Instance) enu.nextElement();
			instance.setWeight(instance.weight() * BobotLama
					/ BobotBaru);
		}
	}


	protected void SMOTEBoostAlgorithm(Instances data) throws Exception {

		Instances datatraining;
		Instances training[] = new Instances[m_Classifier.length];
		double epsilon, Bobot;
		Evaluation evaluasi;
		int indeksnum = data.numInstances();
		Random indeksrandom = new Random(m_Seed);

		
		bobot_votes = new double[m_Classifier.length];
		iterasi_sukses = 0;

		training[iterasi_sukses] = new Instances(data, 0,
				indeksnum);

		for (iterasi_sukses = 0; iterasi_sukses < m_Classifier.length; iterasi_sukses++) {
			if (data_error) {
				System.err.println("Klasifikasi training dengan pengulangan: "
						+ iterasi_sukses);
			}
			
			if (bobot_training < 100) {
				datatraining = bobotkuantil(
						training[iterasi_sukses],
						(double) bobot_training / 100);
			} else {
				datatraining = new Instances(training[iterasi_sukses],
						0, indeksnum);
			}

			// SMOTE
			SMOTE smote = initSMOTE();
			smote.setInputFormat(datatraining);
			datatraining = Filter.useFilter(datatraining, smote);


			if (m_Classifier[iterasi_sukses] instanceof Randomizable)
				((Randomizable) m_Classifier[iterasi_sukses])
						.setSeed(indeksrandom.nextInt());
			m_Classifier[iterasi_sukses].setClassifier(datatraining);

		
			evaluasi = new Evaluation(data);
			evaluasi.evaluateModel(m_Classifier[iterasi_sukses],
					data);
			epsilon = evaluasi.NilaiError();

		
			bobot_votes[iterasi_sukses] = Math.log((1 - epsilon)
					/ epsilon);
			Bobot = (1 - epsilon) / epsilon;

			if (data_error) {
				System.err.println("\tNilai Error = " + epsilon + "  Beta = "
						+ bobot_votes[iterasi_sukses]);
			}

		
			if (iterasi_sukses + 1 < m_Classifier.length) {
				training[iterasi_sukses + 1] = new Instances(
						training[iterasi_sukses], 0, indeksnum);
				setBobot(training[iterasi_sukses + 1], Bobot);
			}
		}
	}
	

	public static void main(String[] argv) {
		runClassifier(new SMOTEBoostAlgorithm(), argv);
	}
}
