import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import binom_test

def make_age_group(df, group=(18,40), name='18-40'):
    """Finds ages within specified range. 
    Adds values to the age group columns of the dataframe with given
    name argument.
    """
    lft = np.where(df.age.astype(int)<group[0], True, False)
    group = np.where((df.age.astype(int)<group[1]) & (~lft), True, False)
    lft = np.where(group, True, lft)
    df.age_group = np.where(group, name, df.age_group)
    return df

def get_confused_groups(df, idx, ids):
    """Gets the descriptors of the predicted class."""
    t_id = df.loc[idx].pred_id
    person = ids[ids.unique_id==t_id]
    sex = list(person.sex)[0]
    race = list(person.race)[0]
    age = list(person.age)[0]
    age_group = list(person.age_group)[0]
    return sex, race, age, age_group

def get_error_rate(df, sex=None, race=None, age_group=None):
    if not sex and not race and not age_group:
        sub = df
    elif sex and not race and not age_group:
        sub = df[df.sex==sex]
    elif race and not sex and not age_group:
        sub = df[df.race==race]
    elif age_group and not sex and not race:
        sub = df[df.age_group==age_group]
    elif age_group and not sex and not race:
        sub = df[df.age_group==age_group]
    elif age_group and sex and not race:
        sub = df[(df.age_group==age_group)&(df.sex==sex)]
    elif age_group and race and not sex:
        sub = df[(df.age_group==age_group)&(df.race==race)]
    elif race and sex and not age_group:
        sub = df[(df.sex==sex)&(df.race==race)]
    elif race and sex and age_group:
        sub = df[(df.age_group==age_group)&(df.race==race)&(df.sex==sex)]
    err = 1 - (len(sub[sub.correct])/len(sub))
    return err

def plot_sex_split(df):
    male = get_error_rate(df, sex='male')
    female = get_error_rate(df, sex='female')
    plt.title('Male vs. Female Error')
    plt.bar(['male', 'female'], [male, female])
    plt.ylabel('error rate %')
    plt.show()
    
def plot_race_split(df):
    asn = get_error_rate(df, race='asian')
    blk = get_error_rate(df, race='black')
    ccn = get_error_rate(df, race='caucasian')
    idn = get_error_rate(df, race='indian')
    hsp = get_error_rate(df, race='hispanic')
    tot = get_error_rate(df)
    plt.title('Error Split on Race')
    plt.bar(['asian', 'black', 'caucasian', 'indian', 'hispanic'], [asn, blk, ccn, idn, hsp])
    plt.axhline(y=tot, label='average error', color='black', linestyle='dotted')
    plt.ylabel('error rate %')
    plt.legend();
    plt.show()
    
def plot_age_split(df):
    e1 = get_error_rate(df, age_group='18-40')
    e2 = get_error_rate(df, age_group='41+')
    plt.title('Error Split on Age')
    plt.bar(['20-39', '40+'], [e1, e2])
    plt.ylabel('error rate %')
    plt.show()
    
def split_group(df, sex=None, race=None, age_group=None):    
    """Creates a plot comparing a select subgroup with the
    total population.
    """
    grp = df
    var1 = 'population'
    var2 = ''
    if age_group:
        grp = grp[grp.age_group==age_group]
        var2 += age_group
    if sex:
        grp = grp[grp.sex==sex]
        var2 = sex + ' ' + var2
    if race:
        grp = grp[grp.race==race]
        var2 = race + ' ' + var2
    pop = df[~df.index.isin(grp.index)]
    if len(pop) == 0 or len(grp) == 0:
        print('No examples in split: ', sex, race, age_group)
    return grp, pop, [var1, var2]

def make_description(sex=None, race=None, age_group=None):
    """Returns a one-string description of arguments."""
    groupname = ''
    if sex:
        groupname += ' ' + sex
    if race:
        groupname += ' ' + race
    if age_group:
        groupname += ' ' + age_group
    return groupname

def plot_confusion(pop, grp, sex=None, race=None, age_group=None):
    """Shows confusion among other and same groups"""
    if race:
        race = ' ' + race
    conf_races = grp.pred_race.unique()
    conf_sexes = grp.pred_sex.unique()
    conf_age_groups = grp.pred_age_group.unique()
    grpnames = []
    conf_rates = []
    for sx in conf_sexes:
        for rc in conf_races:
            for gp in conf_age_groups:
                grpname = make_description(sx, rc, gp)
                conf = grp[~grp.correct]
                conf = len(conf[(conf.pred_sex==sx)&(conf.pred_race==rc)&(conf.pred_age_group==gp)])
                if conf > 0:
                    grpnames.append(grpname)
                    conf_rates.append(conf)
    if len(grpnames) == 1:
        grpnames.append('other (none)')
        conf_rates.append(0)
    groupname = make_description(sex, race, age_group)
    org_df = pd.DataFrame()
    org_df['name'] = grpnames
    org_df['conf_rates'] = conf_rates
    org_df = org_df.sort_values('conf_rates', ascending=False)
    plt.figure(figsize=(12,3))
    plt.title('Confusion classes for {}'.format(groupname))
    if sex:
        sex_logic = np.array([sex in x for x in org_df.name])
    else:
        sex_logic = np.array([True for x in org_df.name])
    if race:
        race_logic = np.array([race in x for x in org_df.name])
    else:
        race_logic = np.array([True for x in org_df.name])
    if age_group:
        age_logic = np.array([age_group in x for x in org_df.name])
    else:
        age_logic = np.array([True for x in org_df.name])
    clr = np.where(sex_logic&race_logic&age_logic, True, False)
    clr = np.where(clr, 'green', 'orange')
    plt.bar(org_df.name, org_df.conf_rates, width=.5, color=clr)
    green_patch = mpatches.Patch(color='green', label='Errors within group')
    orng_patch = mpatches.Patch(color='orange', label='Errors outside group')
    plt.legend(handles=[green_patch, orng_patch])
    plt.xticks(rotation=75)
    return

def plot_inds(ax, pop, grp):
    """Plots a pie chart showing the total number of
    unique individuals as a part of all unique individuals.
    """
    ax.set_title('Number of Unique Individuals')
    pop_unq = len(pop.gt_id.unique())
    grp_unq = len(grp.gt_id.unique())
    ax.pie([pop_unq, grp_unq], labels=[str(pop_unq), str(grp_unq) + ' identities in group'])
    ax.text(-.5, 1.1, str(pop_unq+grp_unq) + ' total identities')
    return ax
    

def plot_custom(df, sex=None, race=None, age_group=None):
    grp, pop, x = split_group(df, sex, race, age_group)
    y = []
    y.append(get_error_rate(pop))
    y.append(get_error_rate(grp))
    y2 = [len(pop), len(grp)]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].set_title('Error Comparison')
    ax[0].bar(x, y)
    ax[1].set_title('Number of Predictions')
    ax[1].bar(x, y2)
    ax[1].text(.85, y2[1]+1000, str(y2[1]), color='black', va='center', fontweight='bold')
    ax[2] = plot_inds(ax[2], pop, grp)
    plt.show();
    plot_confusion(pop, grp, sex, race, age_group)
    plt.show();
    
    
def test_custom(df, sex=None, race=None, age_group=None, verbose=True):
    """Runs a binomial test on the prediction rate"""
    grp, pop, x = split_group(df, sex, race, age_group)
    expected_frequency= 1 - get_error_rate(df)
    p = expected_frequency
    obs = 1 - get_error_rate(grp)
    x = len(grp[grp.correct])
    exp = np.where([True for x in range(len(grp))], expected_frequency, 1-expected_frequency)
    result = binom_test(x, n=len(grp), p=p)
    correction_factor = len(grp)/len(grp.gt_id.unique())
    if verbose:
        msg = 'Binomial test for'
        msg += make_description(sex, race, age_group)
        print(msg + ':')
        print('expected accuracy: ', round(expected_frequency, 3))
        print('observed accuracy: ', round(obs, 3))
        print('p=', '\b'+str(round(result, 5)))
        print('p corrected for number of individuals:', round(result*correction_factor, 5))
    return result, result*correction_factor

def group_gambit(df, sex=None, race=None, age_group=None):
    """Goes through a gambit of plots and analyses."""
    groupname = make_description(sex, race, age_group)
    print('Results for group: ', groupname)
    plot_custom(df, sex, race, age_group)
    result, p_corr = test_custom(df, sex, race, age_group)
    return result, p_corr

def exhaust_splits(df):
    """Exhaustively searches through all combinations and records p values."""
    sexes = df.sex.unique()
    races = df.race.unique()
    ages = df.age_group.unique()
    p_values = pd.DataFrame()
    ps = []
    p_cs = []
    sex_ = []
    race_ = []
    age_ = []
    for sex in sexes:
        p, p_c = test_custom(df, sex=sex, verbose=False)
        ps.append(p)
        p_cs.append(p_c)
        sex_.append(sex)
        race_.append(None)
        age_.append(None)
        for race in races:
            p, p_c = test_custom(df, race=race, verbose=False)
            ps.append(p)
            p_cs.append(p_c)
            race_.append(race)
            sex_.append(None)
            age_.append(None)
            p, p_c = test_custom(df, sex=sex, race=race, verbose=False)
            ps.append(p)
            p_cs.append(p_c)
            sex_.append(sex)
            race_.append(race)
            age_.append(None)
            for age in ages:
                p, p_c = test_custom(df, age_group=age, verbose=False)
                ps.append(p)
                p_cs.append(p_c)
                age_.append(age)
                sex_.append(None)
                race_.append(None)
                p, p_c = test_custom(df, sex=sex, age_group=age, verbose=False)
                ps.append(p)
                p_cs.append(p_c)
                sex_.append(sex)
                age_.append(age)
                race_.append(None)
                p, p_c = test_custom(df, race=race, age_group=age, verbose=False)
                ps.append(p)
                p_cs.append(p_c)
                race_.append(race)
                age_.append(age)
                sex_.append(sex)
                p, p_c = test_custom(df, sex=sex, race=race, age_group=age, verbose=False)
                ps.append(p)
                p_cs.append(p_c)
                race_.append(race)
                sex_.append(sex)
                age_.append(age)

    p_values['p'] = ps
    p_values['p_corrected'] = p_cs
    p_values['sex'] = sex_
    p_values['race'] = race_
    p_values['age_group'] = age_

    p_values = p_values.drop_duplicates()
    return p_values