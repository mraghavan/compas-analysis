from __future__ import division
import csv
import numpy as np
from scipy.io import loadmat, savemat
from sklearn import linear_model
import matplotlib.pyplot as plt

# features = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'is_recid', 'r_charge_degree', 'is_violent_recid', 'vr_charge_degree', 'decile_score', 'v_decile_score', 'two_year_recid']
features = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'decile_score', 'v_decile_score', 'two_year_recid']
int_features = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'is_recid', 'is_violent_recid', 'decile_score', 'v_decile_score', 'two_year_recid']

data = []

def att(feature_name):
    return h_dict[feature_name]

def new_att(feature_name):
    return all_features.index(feature_name)

def make_dict(vals):
    d = {}
    for i, val in enumerate(vals):
        d[val] = i
    return d

def zipd(a, b):
    return {k: v for k,v in zip(a,b)}

def zip_print(a, b):
    for k,v in zip(a,b):
        print k, ":", v

sex_list = ['Male', 'Female']
age_cat_list = ['Less than 25', '25 - 45', 'Greater than 45']
race_list = ['Caucasian', 'African-American', 'Hispanic', 'Asian', 'Native American', 'Other']
charge_degree_list = ['M', 'F']

categorical = {
        'sex' : sex_list,
        'age_cat' : age_cat_list,
        'race' : race_list,
        'c_charge_degree' : charge_degree_list
        }


all_features = []
for feature in features:
    if feature in categorical:
        all_features.extend([feature + ': ' + val for val in categorical[feature]])
    else:
        all_features.append(feature)

def featurize(line):
    def get_feature(attribute):
        return map(lambda e: 1 if e == line[att(attribute)] else 0, categorical[attribute])

    new_line = []
    feature_nums = set(att(feature) for feature in features)
    for i, item in enumerate(line):
        if i not in feature_nums:
            continue
        if i in categorical_nums:
            new_line.extend(get_feature(categorical_nums[i]))
        else:
            new_line.append(int(item))

    return new_line

original_data = []
with open('./compas-scores-two-years.csv') as f:
    compas_reader = csv.reader(f)
    header = compas_reader.next()
    # get rid of duplicate priors, decile score
    del header[11]
    del header[47]
    h_dict = {}
    for ind, head in enumerate(header):
        h_dict[head] = ind
    categorical_nums = {att(feature) : feature for feature in categorical}
    i = 0
    for line in compas_reader:
        del line[11]
        del line[47]
        assert len(line) == len(header)
        # recidivism not within 2 years is still recorded
        assert line[att('is_recid')] >= line[att('two_year_recid')]
        if line[att('days_b_screening_arrest')] == '' or abs(int(line[att('days_b_screening_arrest')])) > 30:
            continue
        if int(line[att('is_recid')]) == -1:
            continue
        if line[att('c_charge_degree')] != 'M' and line[att('c_charge_degree')] != 'F':
            continue
        if line[att('score_text')] == 'N/A':
            continue
        if int(line[att('two_year_recid')]) == 1:
            i += 1
        original_data.append(line)
        line = featurize(line)
        data.append(line)

f_dict = {}
for ind, feat in enumerate(features):
    f_dict[head] = ind

data = np.array(data)
original_data = np.array(original_data)

scores = data[:, new_att('decile_score')]
prohibited = [
        'decile_score',
        'v_decile_score',
        'two_year_recid',
        # 'race: Caucasian',
        # 'race: African-American',
        # 'race: Hispanic',
        # 'race: Asian',
        # 'race: Native American',
        # 'race: Other',
        ]
data = np.delete(data, map(new_att, prohibited), 1)
data = np.hstack((data, np.ones((data.shape[0], 1))))
print 'data shape', data.shape

w = np.linalg.pinv(data.T.dot(data)).dot(data.T.dot(scores))
zip_print([f for f in all_features if not f in prohibited] + ['ones'], w)
pred = data.dot(w)
assert pred.shape == (data.shape[0],)
diff = pred - scores
assert diff.shape == (data.shape[0],)
print np.mean(diff**2)
print np.mean(np.abs(diff))
mean_score = np.mean(scores)
SStot = np.sum((scores - mean_score)**2)
SSres = np.sum(diff**2)
r2 = 1 - SSres/SStot
print 'r^2', r2
print 'max prediction', max(pred), 'min prediction', min(pred)

assert original_data.shape != data.shape
recid_nums = np.zeros((len(race_list), 10))
totals = np.zeros((len(race_list), 10))
for person in original_data:
    score = int(person[att('decile_score')]) - 1
    race = person[att('race')]
    assert int(person[att('two_year_recid')]) in (0,1)
    if int(person[att('two_year_recid')]):
        recid_nums[race_list.index(race), score] += 1
    totals[race_list.index(race), score] += 1
print race_list
recid_rates = recid_nums/totals
print recid_rates

rows = range(len(race_list))
cols = range(10)

fig, ax = plt.subplots()

ax.matshow(recid_rates, cmap='gray')
# plt.xticks(cols, range(1, 11))
# plt.yticks(rows, race_list)
plt.locator_params(axis='x',nbins=20)
ax.set_xticklabels(range(11))
ax.set_yticklabels([0] + race_list)
ax.set_xlabel('Decile score')
# plt.matshow(recid_rates)

x, y = np.meshgrid(cols, rows)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    ax.text(x_val, y_val, '%.2f\n(%d)' % (recid_rates[y_val, x_val], totals[y_val, x_val]), va='center', ha='center', color='red')
fig.subplots_adjust(left=0.2)
plt.savefig('race_score_rates.png')
plt.show()
