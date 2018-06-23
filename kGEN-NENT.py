import tempfile
import tensorflow as tf
import pandas as pd
import numpy as np

###LOGGING SET-UP###
tf.logging.set_verbosity(tf.logging.INFO)

###SMALL DATASETS, NO INCLUSION OF ARG=XREF###
Train_Data2 = './train_data-nent-cg.csv'
Test_Data2 = './test_data-nent-cg.csv'
pred_data = './pred_data-nent.csv'

#9v4
# #feature_columns = ['LmTarget', 'subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2']
#DNN_COLUMNS = ['LmTarget', 'subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2', 'Labels']

#14v4
DNN_COLUMNS = ['nn', 'line', 'subj', 'dobj', 'syn', 'verb', 'iobj', 'obl1', 'obl2', 'Labels']
feature_columns = list(DNN_COLUMNS[:-1])

###DATA INPUT BUILDER###
df_train1 = pd.read_csv(Train_Data2, names=DNN_COLUMNS, skipinitialspace=True)
df_test1 = pd.read_csv(Test_Data2, names=DNN_COLUMNS, skipinitialspace=True)
df_pred = pd.read_csv(pred_data, names=DNN_COLUMNS, skipinitialspace=True)
df_pred['Labels']=int(0)

df_train = pd.concat([df_train1[feature_columns], df_train1['Labels'].astype(int)], axis=1, join='inner')
df_test = pd.concat([df_test1[feature_columns], df_test1['Labels'].astype(int)], axis=1, join='inner')


###SETTING CLASSES AND THE NUMBER OF DNN VARIABLES###
#It's not good enough to simply set the number of classes you might have. We
# really want to have some sort of automated and accurate way of rendering the
# number of classes we're looking at so we can avoid the mistake earlier of
# SAYING we have x-classes, and really having less than that (and thus having
# a loss-value of +5.4 . . . ).

##NOTE: the number of classes must always be n+1
nClasses= int(len(set(df_train['Labels'].values.tolist())))
#all_classes = list(set(df_train['Labels'].values.tolist()))
#S_V_hash_size = int(len(df_train))

#optimal is nD where n = the number of input features.
#optimum drop_out is .4
nD = 7
drop_out_x = .4
hd_units=[100, nClasses]
num_tr_steps=int(len(df_train)*20)
model_dir = [,
        './_models/kGEN-nent/',,
        #tempfile.mkdtemp()
        ]

early_stop=500

###DNN SETUP###
#Herein lies the powerhouse of this system. The following establishes a DNN
# in which the edited information above is passed into a network classifier
# and is then classified to what its 'LmSource' value ought to be.

CATEGORICAL_COLUMNS = feature_columns
LABELS_COLUMN = ['Labels']

def input_fn(df):
        # Creates a dictionary mapping from each continuous feature column name (k) to
        # the values of that column stored in a constant Tensor.
        #continuous_cols = {k: tf.constant(df[k].values)
                                #  for k in CONTINUOUS_COLUMNS}
        # Creates a dictionary mapping from each categorical feature column name (k)
        # to the values of that column stored in a tf.SparseTensor.
        categorical_cols = {k: tf.SparseTensor(
                indices=[[i, 0] for i in range(df[k].size)],
                values=df[k].values,
                dense_shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}
        # Merges the two dictionaries into one.
        feature_cols = dict(categorical_cols.items())
        # Converts the label column into a constant Tensor.
        label = tf.constant(df['Labels'].values.astype(int))
        # Returns the feature columns and the label.
        return feature_cols, label

def train_input_fn():
        return input_fn(df_train)

def eval_input_fn():
        return input_fn(df_test)

def pred_input_fn():
        return input_fn(df_pred)

tref = tf.contrib.layers.sparse_column_with_hash_bucket("nn", hash_bucket_size=int(1000))

subj = tf.contrib.layers.sparse_column_with_hash_bucket("subj", hash_bucket_size=int(15000))

syn = tf.contrib.layers.sparse_column_with_hash_bucket("syn", hash_bucket_size=int(15000))

verb = tf.contrib.layers.sparse_column_with_hash_bucket("verb", hash_bucket_size=int(15000))

obl1 = tf.contrib.layers.sparse_column_with_hash_bucket("obl1", hash_bucket_size=int(15000))

obl2 = tf.contrib.layers.sparse_column_with_hash_bucket("obl2", hash_bucket_size=int(15000))

dobj = tf.contrib.layers.sparse_column_with_hash_bucket("dobj", hash_bucket_size=int(15000))


#0.501
subjxverb = tf.contrib.layers.crossed_column(
	[subj, verb],
	hash_bucket_size=int(1e6),
	combiner='sum')

subjxdobj = tf.contrib.layers.crossed_column(
	[subj, dobj],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.489
dobjxverb = tf.contrib.layers.crossed_column(
	[dobj, verb],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.579
dobjxobl1xobl2 = tf.contrib.layers.crossed_column(
	[dobj, obl1, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

verbxobl1 = tf.contrib.layers.crossed_column(
	[verb, obl1],
	hash_bucket_size=int(1e6),
	combiner='sum')

verbxobl2 = tf.contrib.layers.crossed_column(
	[verb, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

verbxobl1xobl2 = tf.contrib.layers.crossed_column(
	[verb, obl1, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

###MOVEMENT FROM VERB FOCAL TO THETA-ROLE FOCAL MODEL
#The following are new features that could be used in the DNN in order to make
# decisions pertaining to categorization of examples. The percentage above each
# one indicates the total variance across the entiretyof the text. Ideally, DNN
# featues will be around 40-50%, with wide features being low (~10%). To date,
# subj and obj are the only features with that low a variance. The real paradigm
# shift lies in the realization that verb semantics are in fact less useful for
# classification than previoulsy thought with the temporal, toy-data set. From
# here on, we'll be focusing on what would traditionally be the theta-roles
# for categorical differentiation.

#0.442
obl1xobl2 = tf.contrib.layers.crossed_column(
	[obl1, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.449
synxsubj = tf.contrib.layers.crossed_column(
	[syn, subj],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.424
synxdobj = tf.contrib.layers.crossed_column(
	[syn, dobj],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.587
synxobl1 = tf.contrib.layers.crossed_column(
	[syn, obl1],
	hash_bucket_size=int(1e6),
	combiner='sum')

synxverb = tf.contrib.layers.crossed_column(
	[syn, verb],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.739, but very low categorical intersection.
trefxsyn = tf.contrib.layers.crossed_column(
	[syn, tref],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.595
synxobl2 = tf.contrib.layers.crossed_column(
	[syn, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.245
subjxobl1 = tf.contrib.layers.crossed_column(
	[subj, obl1],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.331
subjxobl2 = tf.contrib.layers.crossed_column(
	[subj, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.204
dobjxobl1 = tf.contrib.layers.crossed_column(
	[dobj, obl1],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.307
dobjxobl2 = tf.contrib.layers.crossed_column(
	[dobj, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.369
trefxdobj = tf.contrib.layers.crossed_column(
	[dobj, tref],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.416
trefxsubj = tf.contrib.layers.crossed_column(
	[subj, tref],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.496
trefxobl1 = tf.contrib.layers.crossed_column(
	[obl1, tref],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.479
trefxobl2 = tf.contrib.layers.crossed_column(
	[obl2, tref],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.531
trefxverb = tf.contrib.layers.crossed_column(
	[verb, tref],
	hash_bucket_size=int(1e6),
	combiner='sum')

trefxsubjxdobj = tf.contrib.layers.crossed_column(
	[subj, dobj, tref],
	hash_bucket_size=int(1e6),
	combiner='sum')

trefxverbxdobj = tf.contrib.layers.crossed_column(
	[verb, tref, dobj],
	hash_bucket_size=int(1e6),
	combiner='sum')

trefxverbxsubj = tf.contrib.layers.crossed_column(
	[verb, tref, subj],
	hash_bucket_size=int(1e6),
	combiner='sum')

trefxverbxsubjxdobj = tf.contrib.layers.crossed_column(
	[verb, tref, subj, dobj],
	hash_bucket_size=int(1e6),
	combiner='sum')

trefxobl1xobl2 = tf.contrib.layers.crossed_column(
	[verb, tref, obl1, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

###VALIDATION MONITORING:
#The following are the metrics and set-up for the validation monitor such that
# we can track the progress of the system overtime using Tensorboard.

###OPENING TENSORBOARD:
#Now that we have a way of monitoring everything, we can use TENSORBOARD to
# visually track the progress of KellYGEN as it iterates over the training data.
# To do this, type the following into the terminal:
#             tensorboard --logdir=/tmp/kGEN/
# after doing this, it should prompt you with a port number in yellow-ish.
# Open up any browser and type in http://0.0.0.0:<port number> and it'll open
# tensorboard.

wide_collumns = []

##PREVIOUS deep_columns VALUES
#Similar to the wide_columns, deep_columns are integral to making things work.
# Here are a couple of previous iterations of this set-up.
#       #EX5#   WITH [] wide_columns    .739 acc        dnn_units=[23]
#               tf.contrib.layers.embedding_column(subjxverb, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(dobjxverb, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(dobjxobl1, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(subjxdobj, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(verb, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(syn, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(synxsubj, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(synxdobj, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(synxverb, dimension=nDIMENSION[0]), #TREF x TH-role ID tf.contrib.layers.embedding_column(tref, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(trefxverb, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(trefxsyn, dimension=nDIMENSION[0]),        
#       #EX9#   Dimensions=9D          .781-.790 acc    dnn_units=[23]
#               tf.contrib.layers.embedding_column(subjxverb, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(dobjxverb, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(synxsubj, dimension=nDIMENSION[0]),  tf.contrib.layers.embedding_column(verb, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(obl1, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(syn, dimension=nDIMENSION[0]), #TREF x TH-role ID tf.contrib.layers.embedding_column(tref, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(trefxdobj, dimension=nDIMENSION[0]), tf.contrib.layers.embedding_column(trefxsubj, dimension=nDIMENSION[0]),        
#       #EX10#  Dimensions=10D          .787 acc         dnn_units=[23]
#               tf.contrib.layers.embedding_column(subjxverb, dimension=nD), tf.contrib.layers.embedding_column(dobjxverb, dimension=nD), tf.contrib.layers.embedding_column(synxsubj, dimension=nD), tf.contrib.layers.embedding_column(verb, dimension=nD), tf.contrib.layers.embedding_column(obl1, dimension=nD), tf.contrib.layers.embedding_column(syn, dimension=nD), tf.contrib.layers.embedding_column(obl1xobl2, dimension=nD), #TREF x TH-role ID tf.contrib.layers.embedding_column(tref, dimension=nD), tf.contrib.layers.embedding_column(trefxdobj, dimension=nD), tf.contrib.layers.embedding_column(trefxsubj, dimension=nD),
#       #EX11#  Dimensions=10D          .787 acc        dnn_units=[23]
#               tf.contrib.layers.embedding_column(subjxverb, dimension=nD), tf.contrib.layers.embedding_column(dobjxverb, dimension=nD), tf.contrib.layers.embedding_column(synxsubj, dimension=nD), tf.contrib.layers.embedding_column(verb, dimension=nD), tf.contrib.layers.embedding_column(syn, dimension=nD), tf.contrib.layers.embedding_column(obl1, dimension=nD), tf.contrib.layers.embedding_column(obl2, dimension=nD), #TREF x TH-role ID tf.contrib.layers.embedding_column(tref, dimension=nD), tf.contrib.layers.embedding_column(trefxdobj, dimension=nD), tf.contrib.layers.embedding_column(trefxsubj, dimension=nD),
#       #EX12#  Dimensions=9D           .790 acc        dnn_units=[23]
#               tf.contrib.layers.embedding_column(subjxverb, dimension=nD), tf.contrib.layers.embedding_column(dobjxverb, dimension=nD), tf.contrib.layers.embedding_column(synxsubj, dimension=nD), tf.contrib.layers.embedding_column(verb, dimension=nD), tf.contrib.layers.embedding_column(syn, dimension=nD), tf.contrib.layers.embedding_column(obl1, dimension=nD), tf.contrib.layers.embedding_column(obl2, dimension=nD), #TREF x TH-role ID tf.contrib.layers.embedding_column(trefxdobj, dimension=nD), tf.contrib.layers.embedding_column(trefxsubj, dimension=nD),
#       #EX13#  Dimanesions=
#               #Frame-Semantic ID tf.contrib.layers.embedding_column(subjxverb, dimension=nD), tf.contrib.layers.embedding_column(dobjxverb, dimension=nD), tf.contrib.layers.embedding_column(dobjxobl2, dimension=nD), tf.contrib.layers.embedding_column(synxsubj, dimension=nD), tf.contrib.layers.embedding_column(obl1xobl2, dimension=nD), #Lexical Reference Points tf.contrib.layers.embedding_column(subj, dimension=nD), tf.contrib.layers.embedding_column(dobj, dimension=nD), tf.contrib.layers.embedding_column(verb, dimension=nD), tf.contrib.layers.embedding_column(syn, dimension=nD), tf.contrib.layers.embedding_column(obl1, dimension=nD), tf.contrib.layers.embedding_column(obl2, dimension=nD), #TREF x TH-role ID tf.contrib.layers.embedding_column(tref, dimension=nD), tf.contrib.layers.embedding_column(trefxdobj, dimension=nD), tf.contrib.layers.embedding_column(trefxsubj, dimension=nD), tf.contrib.layers.embedding_column(trefxobl2, dimension=nD),
#       #EX16#  Dimensions=12D          .787 acc
#               tf.contrib.layers.embedding_column(subjxverb, dimension=nD), tf.contrib.layers.embedding_column(dobjxverb, dimension=nD), tf.contrib.layers.embedding_column(synxsubj, dimension=nD), tf.contrib.layers.embedding_column(obl1xobl2, dimension=nD), tf.contrib.layers.embedding_column(verb, dimension=nD), tf.contrib.layers.embedding_column(syn, dimension=nD), tf.contrib.layers.embedding_column(subj, dimension=nD), tf.contrib.layers.embedding_column(dobj, dimension=nD), tf.contrib.layers.embedding_column(obl1, dimension=nD), tf.contrib.layers.embedding_column(tref, dimension=nD), tf.contrib.layers.embedding_column(trefxdobj, dimension=nD), tf.contrib.layers.embedding_column(trefxsubj, dimension=nD),
#       #EX17#  Dimensions=9D           .789-.7983 acc
#               tf.contrib.layers.embedding_column(subjxverb, dimension=nD), tf.contrib.layers.embedding_column(dobjxverb, dimension=nD), tf.contrib.layers.embedding_column(synxsubj, dimension=nD), tf.contrib.layers.embedding_column(obl1xobl2, dimension=nD), tf.contrib.layers.embedding_column(verb, dimension=nD), tf.contrib.layers.embedding_column(syn, dimension=nD), tf.contrib.layers.embedding_column(subj, dimension=nD), tf.contrib.layers.embedding_column(obl1, dimension=nD), tf.contrib.layers.embedding_column(tref, dimension=nD),
#       #EX18#  Dimensions=10D          .792 acc
#               #Frame-Semantic Reference (+7) tf.contrib.layers.embedding_column(synxsubj, dimension=nD), tf.contrib.layers.embedding_column(obl1xobl2, dimension=nD), tf.contrib.layers.embedding_column(verb, dimension=nD), tf.contrib.layers.embedding_column(syn, dimension=nD), tf.contrib.layers.embedding_column(subj, dimension=nD), tf.contrib.layers.embedding_column(dobj, dimension=nD), tf.contrib.layers.embedding_column(obl1, dimension=nD), #TREF RELATED ELEMENTS (+3) tf.contrib.layers.embedding_column(tref, dimension=nD), tf.contrib.layers.embedding_column(trefxdobj, dimension=nD), tf.contrib.layers.embedding_column(trefxsubj, dimension=nD),
#       #EX19#  Dimensions=7D           .778 acc
#               tf.contrib.layers.embedding_column(tref, dimension=nD), tf.contrib.layers.embedding_column(obl1, dimension=nD), tf.contrib.layers.embedding_column(syn, dimension=nD), tf.contrib.layers.embedding_column(verb, dimension=nD), tf.contrib.layers.embedding_column(trefxdobj, dimension=nD), tf.contrib.layers.embedding_column(trefxsubj, dimension=nD), tf.contrib.layers.embedding_column(trefxobl2, dimension=nD), tf.contrib.layers.embedding_column(trefxverb, dimension=nD),
#       #EX20#  Dimensions=11D          .798            dnn_units=[77]  dropout=.4
#               #Frame-Semantic Reference (+6) tf.contrib.layers.embedding_column(verb, dimension=nD), tf.contrib.layers.embedding_column(syn, dimension=nD), tf.contrib.layers.embedding_column(subj, dimension=nD), tf.contrib.layers.embedding_column(dobj, dimension=nD), tf.contrib.layers.embedding_column(obl1, dimension=nD), tf.contrib.layers.embedding_column(obl2, dimension=nD), #TREF RELATED ELEMENTS (+5) tf.contrib.layers.embedding_column(tref, dimension=nD), tf.contrib.layers.embedding_column(trefxdobj, dimension=nD), tf.contrib.layers.embedding_column(trefxsubj, dimension=nD), tf.contrib.layers.embedding_column(trefxverb, dimension=nD), tf.contrib.layers.embedding_column(trefxobl2, dimension=nD),
#       #EX21#  Dimensions=12D          .801 acc        dnn_units=[77]  dropout=.4
#               #Frame-Semantic Reference (+6) tf.contrib.layers.embedding_column(verb, dimension=nD), tf.contrib.layers.embedding_column(syn, dimension=nD), tf.contrib.layers.embedding_column(subj, dimension=nD), tf.contrib.layers.embedding_column(dobj, dimension=nD), tf.contrib.layers.embedding_column(obl1, dimension=nD), tf.contrib.layers.embedding_column(obl2, dimension=nD), #TREF RELATED ELEMENTS (+6) tf.contrib.layers.embedding_column(tref, dimension=nD), tf.contrib.layers.embedding_column(trefxdobj, dimension=nD), tf.contrib.layers.embedding_column(trefxsubj, dimension=nD), tf.contrib.layers.embedding_column(trefxobl2, dimension=nD), tf.contrib.layers.embedding_column(trefxverbxdobj, dimension=nD), tf.contrib.layers.embedding_column(trefxverbxsubj, dimension=nD),
#       #EX22#  Dimensions=13D          .804 acc        dnn_units=[77]  dropout=.4      steps=31000
#               #Frame-Semantic Reference (+6) tf.contrib.layers.embedding_column(verb, dimension=nD), tf.contrib.layers.embedding_column(syn, dimension=nD), tf.contrib.layers.embedding_column(subj, dimension=nD), tf.contrib.layers.embedding_column(dobj, dimension=nD), tf.contrib.layers.embedding_column(obl1, dimension=nD), tf.contrib.layers.embedding_column(obl2, dimension=nD), #TREF RELATED ELEMENTS (+6) tf.contrib.layers.embedding_column(tref, dimension=nD), tf.contrib.layers.embedding_column(trefxdobj, dimension=nD), tf.contrib.layers.embedding_column(trefxsubj, dimension=nD), tf.contrib.layers.embedding_column(trefxobl2, dimension=nD), tf.contrib.layers.embedding_column(trefxverbxdobj, dimension=nD), tf.contrib.layers.embedding_column(trefxverbxsubj, dimension=nD), tf.contrib.layers.embedding_column(trefxobl1xobl2, dimension=nD),



deep_columns = [
        ###GEN NOTE: All elements not listed as TEST-SET are tried and true,
        # and should be left well enough alone in its current arrangement.

        #Frame-Semantic Reference (+6)
        tf.contrib.layers.embedding_column(verb, dimension=nD),
        tf.contrib.layers.embedding_column(syn, dimension=nD),
        #tf.contrib.layers.embedding_column(subj, dimension=nD),
        #tf.contrib.layers.embedding_column(dobj, dimension=nD),
        tf.contrib.layers.embedding_column(obl1, dimension=nD),
        #tf.contrib.layers.embedding_column(obl2, dimension=nD),

        #TREF RELATED ELEMENTS (+7)
        tf.contrib.layers.embedding_column(tref, dimension=nD),
        #tf.contrib.layers.embedding_column(trefxdobj, dimension=nD),
        #tf.contrib.layers.embedding_column(trefxsubj, dimension=nD),
        #tf.contrib.layers.embedding_column(trefxverb, dimension=nD),
        tf.contrib.layers.embedding_column(dobjxverb, dimension=nD),
        tf.contrib.layers.embedding_column(subjxverb, dimension=nD),
        tf.contrib.layers.embedding_column(obl1xobl2, dimension=nD),

        #TEST-SET (+1 each)
        #tf.contrib.layers.embedding_column(synxsubj, dimension=nD),
        ]


validation_metrics = {
        #The below is the best bet to run accuracy in here, but we need to
        # somehow run labels as a full-blown tensor of some sort.
        'accuracy': tf.contrib.metrics.streaming_accuracy,
        'precision': tf.contrib.metrics.streaming_precision,
        'recall': tf.contrib.metrics.streaming_recall
        }

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        #df_test[feature_columns].values,
        #df_test['Labels'].values,
        input_fn=eval_input_fn,
        every_n_steps=len(df_train),
        metrics=validation_metrics,
        early_stopping_rounds=early_stop,
        early_stopping_metric='loss',
        early_stopping_metric_minimize=False
        )

m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir[0],
        linear_feature_columns=wide_collumns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hd_units,
        n_classes=nClasses,
        dnn_dropout=drop_out_x,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=5),
        fix_global_step_increment_bug=True
        )

pred_m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir[0],
        linear_feature_columns=wide_collumns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hd_units,
        n_classes=nClasses,
        #config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10),
        fix_global_step_increment_bug=True
        )





#####
###FUNCTION: TRAINING THE MODEL
#####
class dnn:
        def train_model(dropout,  train_steps=num_tr_steps, calc_results=True, resultsteps=10):
                f_m = tf.contrib.learn.DNNLinearCombinedClassifier(
                        model_dir=model_dir[0],
                        linear_feature_columns=wide_collumns,
                        dnn_feature_columns=deep_columns,
                        dnn_hidden_units=hd_units,
                        n_classes=nClasses,
                        dnn_dropout=dropout,
                        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=40),
                        fix_global_step_increment_bug=True)
                f_m.fit(input_fn=train_input_fn, steps=train_steps)
                if calc_results==True:
                        results = pred_m.evaluate(input_fn=eval_input_fn, steps=resultsteps)
                        return results

        def predict():
                predictions=pred_m.predict_classes(input_fn=pred_input_fn)
                entity=list(zip(df_pred['nn'].values.tolist(), predictions))
                return entity


#####
##IMPLEMENTATION
#####



##PRINT LEN OF TEST RUNS##
#print(len(df_train), ' iterations per epoch \n')
#print(num_tr_steps, ' steps to be trained. \n')
#print(nClasses, 'classes to be run \n')
pause=input('Press enter to start. \n')

##TRAIN & TEST MODEL##
#x=dnn.train_model(drop_out_x, num_tr_steps)
#print(x)

pred=dnn.predict()

