###############################################################################################
# our imports:
###############################################################################################


import shap
import lime
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from imageio import imread
from sklearn.impute import SimpleImputer
from sklearn import preprocessing, linear_model, model_selection, metrics
from sklearn.model_selection import train_test_split
from lime import lime_image, submodular_pick
from skimage.segmentation import mark_boundaries
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2


# just to ignore some unnecessary and annoying warnings like deprecations and all:
warnings.filterwarnings('ignore')

# taken from https://www.youtube.com/watch?v=Z2kfLs2Dwqw&ab_channel=bsaldivar%3ADatascience
shap.explainers._deep.deep_tf.op_handlers['AddV2'] = shap.explainers._deep.deep_tf.passthrough


###############################################################################################
# our imports:
###############################################################################################



###############################################################################################
# question 1 - part (b):
###############################################################################################


# load the dataset:
df = pd.read_csv('Life Expectancy Data.csv')

print(df.head())
print(f'{df.columns = }\n')
print(f'{df.shape = }\n')

# make a new feature called continents:
# inspired by https://stackoverflow.com/questions/55910004/get-continent-name-from-country-using-pycountry

def convert_to_common_names(country):
	# since, "pycountry_convert" package couldn't recognize which continent this country is in:
	if country == 'Timor-Leste':
		return 'Asia'
	else:
		return continents[(country_alpha2_to_continent_code(country_name_to_country_alpha2(country)))]

# names of some countries have parentheses, like: "Bolivia (Plurinational State of)", "Iran (Islamic Republic of)"...
# ...we have to remove them:
df['Country'] = df['Country'].str.replace(r' \(.*\)', '')

# modernizing or converting some country names to their common form:
df['Country'] = df['Country'].str.replace('Republic of Korea', 'South Korea')
df['Country'] = df['Country'].str.replace('Democratic People\'s South Korea', 'South Korea')
df['Country'] = df['Country'].str.replace('The former Yugoslav republic of Macedonia', 'North Macedonia')

continents = {'NA': 'North America', 'SA': 'South America', 'AS': 'Asia', 'OC': 'Australia', 'AF': 'Africa', 'EU': 'Europe'}

# applying a function on the dataframe to create a new column called "Continent":
# we use apply method since it is much faster than for-loop.
df['Continent'] = df['Country'].apply(lambda x: convert_to_common_names(x))

print(df[['Country', 'Continent']])

# replace the null or missing values with mean values of the feature:
# taken from: https://www.kaggle.com/code/varunsaikanuri/life-expectancy-visualization

print('Before imputing missing values:\n', df.isnull().sum())

imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None)

df['Life expectancy '] = imputer.fit_transform(df[['Life expectancy ']])
df['GDP'] = imputer.fit_transform(df[['GDP']])
df['Population'] = imputer.fit_transform(df[['Population']])
df['Adult Mortality'] = imputer.fit_transform(df[['Adult Mortality']])
df['Alcohol'] = imputer.fit_transform(df[['Alcohol']])
df['Hepatitis B'] = imputer.fit_transform(df[['Hepatitis B']])
df[' BMI '] = imputer.fit_transform(df[[' BMI ']])
df['Polio'] = imputer.fit_transform(df[['Polio']])
df['Total expenditure'] = imputer.fit_transform(df[['Total expenditure']])
df['Diphtheria '] = imputer.fit_transform(df[['Diphtheria ']])
df[' thinness  1-19 years'] = imputer.fit_transform(df[[' thinness  1-19 years']])
df[' thinness 5-9 years'] = imputer.fit_transform(df[[' thinness 5-9 years']])
df['Income composition of resources'] = imputer.fit_transform(df[['Income composition of resources']])
df['Schooling'] = imputer.fit_transform(df[['Schooling']])

print('After imputing missing values:\n', df.isnull().sum())

# removing this feature since it has negative effect on training:
# we actually compared the accuracy scores and saw that, this feature is better to be removed.
# but since the question asks us to observe SHAP values on all features, we don't remove this feature.
#df.drop('Year', inplace=True, axis=1)

# splitting the data into x and y:
x = df.drop('Life expectancy ', axis=1)
y = df[['Life expectancy ']]

# split the data into training and testing sets, stratifying on the 'Continent' feature
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=df['Continent'], random_state=42)

# iterate over the continents present in the testing data:
# we want to make sure that we have at least 3 unique countries from each continent in test data.
for continent in x_test['Continent'].unique():
    # get a list of all countries in the testing data for this continent:
    countries = x_test[x_test['Continent'] == continent]['Country'].unique()

    # ensure that there are at least 3 unique countries in the testing data for this continent:
    # if it throws an error, it means there are not at least 3 unique countries in each continent.
    assert len(countries) >= 3, f'At least 3 unique countries from {continent} must be in the testing data'

# in order to solve last part, we have to have indices of datapoints for each continent.
# we will take two datapoints for each continent.
# later, we will use these indices to produce new array and solve the last paragraph of this section of the question.
# the reason for doing so is that since we do standardization, scaling and such, there's no way of...
# ...telling which datapoint belongs to which continent or which country.

# the key is the name of the continents and the value for each key is a list containing indices...
# ...of the countries belonging to that cotinent.
continents_indices_for_df = {value: [] for _, value in continents.items()}
continents_indices_for_x_test = {value: [] for _, value in continents.items()}

while True:
	# from each continent we take two samples:
	new_df = x_test.groupby('Continent').apply(lambda x: x.sample(n=2))#.reset_index(drop=True)

	# we have to make sure that these two samples are from different countries:
	if len(new_df['Country'].unique()) == 2 * len(continents_indices_for_df.keys()):
		print(new_df)

		# we just simply take indices from each continent and assign it to our dictionary:
		# i know that the codes do actually intimidating, but they are actually working.
		# the point is, for "continents_indices_for_df" i have to have the original index...
		# ...from the original dataset, df, so that i can access country name...
		# ...and as for "continents_indices_for_x_test", i need the reset (past tense) index of this...
		# ...array so that i can actually use the contents correctly to plot the SHAP values.
		for continent in new_df['Continent'].unique():
			continents_indices_for_df[continent] = [int(item[1]) for item in new_df[new_df['Continent'] == continent].index.values]
			continents_indices_for_x_test[continent] = [int(x_test.reset_index()[x_test.reset_index()['index'] == int(item[1])].index.values) for item in new_df[new_df['Continent'] == continent].index.values]
		
		break

print('Continents indices for original df:\n', continents_indices_for_df)
print('\n')
print('Continents indices for test df:\n', continents_indices_for_x_test)

# to prove that items from x_test and df with their own indices point to actually same datapoint:
# i pick randomly, you are free to try it for yourself.
# you won't see "Life expectancy" column in x_test:
print('From original df:\n', df.iloc[continents_indices_for_df['Asia'][0]])
print('\nFrom test df:\n', x_test.iloc[continents_indices_for_x_test['Asia'][0]])

# removing Continent feature after we're done with it:
x_train.drop('Continent', inplace=True, axis=1)
x_test.drop('Continent', inplace=True, axis=1)

# turning text strings into categorical values:
l_encoder= preprocessing.LabelEncoder()
x_train.loc[:, 'Status'] = l_encoder.fit_transform(x_train.loc[:, 'Status'])

l_encoder= preprocessing.LabelEncoder()
x_test.loc[:, 'Status'] = l_encoder.fit_transform(x_test.loc[:, 'Status'])

l_encoder= preprocessing.LabelEncoder()
x_train.loc[:, 'Country'] = l_encoder.fit_transform(x_train.loc[:, 'Country'])

l_encoder= preprocessing.LabelEncoder()
x_test.loc[:, 'Country'] = l_encoder.fit_transform(x_test.loc[:, 'Country'])

'''
# standardizing and scaling the train and test data:
# inspired by https://github.com/BatuhanSeremet/Life_Expectancy-Regression/blob/main/Reg_Life_Expectancy.ipynb

scaler = preprocessing.StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
'''

# MinMax scaling the train and test data:
# inspired by https://www.kaggle.com/code/mohamedsalemmohamed/life-expectancy-prediction-with-nn

scaler = preprocessing.MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# to convert our data to proper numpy arrays for later classification:
x_train = np.asarray(x_train).astype(np.float64)
y_train = np.asarray(y_train).astype(np.float64)
x_test = np.asarray(x_test).astype(np.float64)
y_test = np.asarray(y_test).astype(np.float64)

print(f'{x_train.shape = }')
print(f'{y_train.shape = }')
print(f'{x_test.shape = }')
print(f'{y_test.shape = }')

# we are building the architecture of our model. we use Model and Functional API:

'''
inputs = tf.keras.layers.Input(shape=x_train.shape[1])

features = tf.keras.layers.Dense(32)(inputs)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.activations.selu(features)
#features = tf.keras.layers.Dropout(0.05)(features)

features = tf.keras.layers.Dense(64)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.activations.selu(features)
#features = tf.keras.layers.Dropout(0.05)(features)

features = tf.keras.layers.Dense(128)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.activations.selu(features)
#features = tf.keras.layers.Dropout(0.05)(features)

features = tf.keras.layers.Dense(256)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.activations.selu(features)
#features = tf.keras.layers.Dropout(0.1)(features)

features = tf.keras.layers.Dense(512)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.activations.selu(features)
#features = tf.keras.layers.Dropout(0.1)(features)

features = tf.keras.layers.Dense(1024)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.activations.selu(features)
#features = tf.keras.layers.Dropout(0.1)(features)

features = tf.keras.layers.Dense(512)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.activations.selu(features)
#features = tf.keras.layers.Dropout(0.1)(features)

features = tf.keras.layers.Dense(256)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.activations.selu(features)
#features = tf.keras.layers.Dropout(0.1)(features)

features = tf.keras.layers.Dense(128)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.activations.selu(features)
#features = tf.keras.layers.Dropout(0.1)(features)

features = tf.keras.layers.Dense(64)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.activations.selu(features)
#features = tf.keras.layers.Dropout(0.15)(features)

features = tf.keras.layers.Dense(32)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.activations.selu(features)
#features = tf.keras.layers.Dropout(0.1)(features)

outputs = tf.keras.layers.Dense(1, activation='linear')(features)
'''

inputs = tf.keras.layers.Input(shape=x_train.shape[1])

features = tf.keras.layers.Dense(32)(inputs)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.layers.ReLU()(features)
#features = tf.keras.layers.Dropout(0.05)(features)

features = tf.keras.layers.Dense(64)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.layers.ReLU()(features)
#features = tf.keras.layers.Dropout(0.05)(features)

features = tf.keras.layers.Dense(128)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.layers.ReLU()(features)
#features = tf.keras.layers.Dropout(0.05)(features)

features = tf.keras.layers.Dense(256)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.layers.ReLU()(features)
#features = tf.keras.layers.Dropout(0.1)(features)

features = tf.keras.layers.Dense(512)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.layers.ReLU()(features)
#features = tf.keras.layers.Dropout(0.1)(features)

features = tf.keras.layers.Dense(1024)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.layers.ReLU()(features)
#features = tf.keras.layers.Dropout(0.1)(features)

features = tf.keras.layers.Dense(512)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.layers.ReLU()(features)
#features = tf.keras.layers.Dropout(0.1)(features)

features = tf.keras.layers.Dense(256)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.layers.ReLU()(features)
#features = tf.keras.layers.Dropout(0.1)(features)

features = tf.keras.layers.Dense(128)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.layers.ReLU()(features)
#features = tf.keras.layers.Dropout(0.1)(features)

features = tf.keras.layers.Dense(64)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.layers.ReLU()(features)
#features = tf.keras.layers.Dropout(0.15)(features)

features = tf.keras.layers.Dense(32)(features)
features = tf.keras.layers.BatchNormalization()(features)
features = tf.keras.layers.ReLU()(features)
#features = tf.keras.layers.Dropout(0.1)(features)

outputs = tf.keras.layers.Dense(1, activation='linear')(features)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# to see the architecture and the layers of our model:
model.summary()

# compile the model:
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

# train the model:
history = model.fit(x_train, y_train, epochs=200, batch_size=64, validation_split=0.1)

# plot the training progress:
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Evaluation of the Linear Regression Model', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.legend(loc='best', fontsize=20)
plt.show()

# evaluate the model on test-set:
model.evaluate(x=x_test, y=y_test)

# now onto SHAP values:

# first, kernelshap method:

# since we don't have the enough processing power to process all train-set data...
# ...we just take 500 datapoints from it.
# inspired by https://towardsdatascience.com/deep-learning-model-interpretation-using-shap-a21786e91d16

#background = shap.kmeans(x_train, 500)
background = x_train[np.random.choice(x_train.shape[0], 500, replace=False)]

# we will create the objects for our explainer and SHAP method with the help of our model...
# ...and train data (all or part of the data) and try to explain based upon the features:

explainer_kernel_shap = shap.KernelExplainer(model.predict, background)
shap_values_kernel_shap = explainer_kernel_shap.shap_values(x_test)

# now, we will plot the summary plot for kernelshap method:
shap.summary_plot(shap_values_kernel_shap[0], x_test, feature_names=x.columns, plot_type='bar')

# second, deepshap method:

# some parts are inspired from https://snyk.io/advisor/python/shap/functions/shap.DeepExplainer and...
# ...also https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/deep_explainer/DeepExplainer%20Genomics%20Example.html
# "check_additivity=False" is inspired by https://github.com/slundberg/shap/issues/930

#explainer_deep_shap = shap.DeepExplainer(model, background)
#explainer_deep_shap = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
explainer_deep_shap = shap.DeepExplainer((model.input, model.output), background)

shap_values_deep_shap = explainer_deep_shap.shap_values(x_test, check_additivity=False)

# now, we will plot the summary plot for deepshap method:
shap.summary_plot(shap_values_deep_shap[0], x_test, feature_names=x.columns, plot_type='bar')

# now, onto the last part and drawing the force_plot for each of the methods:

'''
# just for testing...

# first for kernelshap method:

shap_values_kernel_shap = explainer_kernel_shap.shap_values(x_test[1, :])

shap.force_plot(explainer_kernel_shap.expected_value, shap_values_kernel_shap[0], x_test[1, :],
				matplotlib=True, show=True, feature_names=x.columns, text_rotation=45)

plt.show()

# then, for deepshap method:

shap.force_plot(explainer_deep_shap.expected_value.numpy(), shap_values_deep_shap[0][1], x_test[1, :],
				matplotlib=True, show=True, feature_names=x.columns, text_rotation=45)

plt.show()
'''

# we forgot to delete this feature from "x":
x.drop('Continent', inplace=True, axis=1)

for continent in continents_indices_for_x_test.keys():
    counter = 0
    #print(continent)

    for index in continents_indices_for_x_test[continent]:
        country = df.loc[continents_indices_for_df[continent][counter], 'Country']
        index_df = continents_indices_for_df[continent][counter]
        
        #print(country)
        #print(index)
        # first for kernelshap method:

        # we just perform on only one single datapoint from test-set:
        shap_values_kernel_shap = explainer_kernel_shap.shap_values(x_test[index, :])

        # we just plot and show on only one single datapoint from test-set:
        # inspired by https://www.youtube.com/watch?v=Z2kfLs2Dwqw&ab_channel=bsaldivar%3ADatascience
        shap.force_plot(explainer_kernel_shap.expected_value, shap_values_kernel_shap[0], x_test[index, :],
                        matplotlib=True, show=False, feature_names=x.columns, text_rotation=45)

        plt.title(f'KernelSHAP for continent "{continent}", country {country}, index "{index_df}" from original dataset', y=-0.9)
        plt.show()

        # then, for deepshap method:

        # we just plot and show on only one single datapoint from test-set:
        # inspired by https://www.youtube.com/watch?v=Z2kfLs2Dwqw&ab_channel=bsaldivar%3ADatascience
        # np.newaxis is for add a dimension to the first, meaning, instead of having only one single array...
        # ...we want to have an array which has that single array.
        # dimesnions we have: (21,) -> dimensions we will get after np.newaxis: (1, 21)
        shap_values_deep_shap = explainer_deep_shap.shap_values(x_test[index, :][np.newaxis, ...], check_additivity=False)

        shap.force_plot(explainer_deep_shap.expected_value.numpy(), shap_values_deep_shap[0], x_test[index, :],
                        matplotlib=True, show=False, feature_names=x.columns, text_rotation=45)

        plt.title(f'DeepSHAP for continent "{continent}", country "{country}", index "{index_df}" from original dataset', y=-0.92)
        plt.show()

        counter += 1


###############################################################################################
# question 1 - part (b):
###############################################################################################



###############################################################################################
# question 2 - part (a):
###############################################################################################


# load the model with the weights trained on imagenet dataset:
model = MobileNetV2(weights='imagenet')
#model.summary()


###############################################################################################
# question 2 - part (a):
###############################################################################################



###############################################################################################
# question 2 - part (b):
###############################################################################################


# for each part, i create a function so that when i want to try other pictures as well...
# ...i wouldn't rewrite and repeat the codes again and again and again...
# ...just like these comments i'm, shamelessly repeating. ./

def q2_part_b(model_temp, image_name, extension):
	# first read the image, resize it, and save it:
	image = Image.open(f'images/q4 images/{image_name}.{extension}')
	new_image = image.resize((224, 224))
	new_image.save(f'images/q4 images/{image_name} resized.{extension}')

	# now, we will read the image and try to have the model predict the probabilities for the image:
	# inspired by https://pythontutorials.eu/deep-learning/image-classification/

	images = np.empty((1, 224, 224, 3))
	images[0] = imread(f'images/q4 images/{image_name} resized.{extension}')

	# we have to preprocess the image so that the model wouldn't throw errors:
	images = preprocess_input(images)

	# have the model predict some labels for that image:
	predictions = model.predict(images)
	print('Shape: {}'.format(predictions.shape))

	# show the top 5 predictions:
	for name, desc, score in decode_predictions(predictions)[0]:
	    print('- {} ({:.2f}%%)'.format(desc, 100 * score))

	return images

images = q2_part_b(model, 'Komondor-standing-in-the-park', 'jpg') # (contains Komondor)


###############################################################################################
# question 2 - part (b):
###############################################################################################



###############################################################################################
# question 2 - part (c):
###############################################################################################


# for each part, i create a function so that when i want to try other pictures as well...
# ...i wouldn't rewrite and repeat the codes again and again and again...
# ...just like these comments i'm, shamelessly repeating. ./

def q2_part_c(images_temp, model_temp):
	# we will create an object of the lime image explainer and then explain the model with that technique:
	# this function is inspired by the documentation itself.

	explainer = lime_image.LimeImageExplainer()
	exp_temp = explainer.explain_instance(images_temp[0], model_temp.predict, top_labels=5, hide_color=0, num_samples=1000)

	return exp_temp

exp = q2_part_c(images, model)


###############################################################################################
# question 2 - part (c):
###############################################################################################



###############################################################################################
# question 2 - part (d):
###############################################################################################


# for each part, i create a function so that when i want to try other pictures as well...
# ...i wouldn't rewrite and repeat the codes again and again and again...
# ...just like these comments i'm, shamelessly repeating. ./

def q2_part_d(exp_temp, exp_class, weight=0.1):
	# this is a method to display and highlight super-pixels used by the black-box model to make predictions:
	# this function is inspired by https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image ...
	# ...and also https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20images.html ...
	# ...also shout-out to https://towardsdatascience.com/how-to-explain-image-classifiers-using-lime-e364097335b4

    image, mask = exp_temp.get_image_and_mask(exp_class, positive_only=True, num_features=6, 
    										  hide_rest=False, min_weight=weight)

    # now, we will plot and show the boundaries on the image:
    plt.imshow(mark_boundaries(image, mask))
    plt.axis('off')
    plt.show()

q2_part_d(exp, exp.top_labels[0])


###############################################################################################
# question 2 - part (d):
###############################################################################################



###############################################################################################
# question 2 - part (e):
###############################################################################################


# for each part, i create a function so that when i want to try other pictures as well...
# ...i wouldn't rewrite and repeat the codes again and again and again...
# ...just like these comments i'm, shamelessly repeating. ./

def q2_part_e(exp_temp):
	# this function is inspired by https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb

	# the parameters are set in a way so that we can produce the resulting images in a correct way...
	# ...especially "positive_only" and "hide_rest" features.
	image, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, num_features=6, hide_rest=False)
	
	# now we will plot and show:
	plt.imshow(mark_boundaries(image / 2 + 0.5, mask))
	plt.axis('off')
	plt.show()

q2_part_e(exp)


###############################################################################################
# question 2 - part (e):
###############################################################################################



###############################################################################################
# question 2 - part (f):
###############################################################################################

# for each part, i create a function so that when i want to try other pictures as well...
# ...i wouldn't rewrite and repeat the codes again and again and again...
# ...just like these comments i'm, shamelessly repeating. ./

def q2_part_f(exp_temp):
	# this function is inspired by https://medium.com/swlh/immediately-understand-lime-for-ml-model-explanation-part-2-35e4fe9c8264

	# select the same class explained on the figures above:
	index =  exp_temp.top_labels[0]

	# map each explanation weight to the corresponding superpixel:
	dict_heatmap = dict(exp_temp.local_exp[index])
	heatmap = np.vectorize(dict_heatmap.get)(exp_temp.segments)

	# now we will plot and show:
	plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
	plt.colorbar()
	plt.show()

q2_part_f(exp)

###############################################################################################
# question 2 - part (f):
###############################################################################################



###############################################################################################
# question 2 - part (g):
###############################################################################################


# now, we will try all those parts for two other images... we will use the functions we created above:

# image #2: (Siamese cat and Persian cat)

print('Part (b):')
images = q2_part_b(model, 'hqdefault', 'jpg')

print('\nPart (c):')
exp = q2_part_c(images, model)

print('\nPart (d):')
q2_part_d(exp, exp.top_labels[0])

print('\nPart (e):')
q2_part_e(exp)

print('\nPart (f):')
q2_part_f(exp)

# image #3: (French bulldog and Persian cat)

print('Part (b):')
images = q2_part_b(model, 'dok-01-bk1108-01p', 'jpg')

print('\nPart (c):')
exp = q2_part_c(images, model)

print('\nPart (d):')
q2_part_d(exp, exp.top_labels[0])

print('\nPart (e):')
q2_part_e(exp)

print('\nPart (f):')
q2_part_f(exp)


###############################################################################################
# question 2 - part (g):
###############################################################################################