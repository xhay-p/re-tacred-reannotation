import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import pickle

from utils import helper


# embeddings: num_of_sentences*dim_of sentences
# y: target relations
# labels: relations
def get_cmap(n, name='nipy_spectral'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def tsne_plot(embeddings, target, global_step):
    embeddings = embeddings.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    labels = ['no_relation', 'per:title', 'org:top_members/employees', 'per:employee_of', 'org:alternate_names', 'org:country_of_headquarters', 'per:countries_of_residence', 'org:city_of_headquarters', 'per:cities_of_residence', 'per:age', 'per:stateorprovinces_of_residence', 'per:origin', 'org:subsidiaries', 'org:parents', 'per:spouse', 'org:stateorprovince_of_headquarters', 'per:children', 'per:other_family', 'per:alternate_names', 'org:members', 'per:siblings', 'per:schools_attended', 'per:parents', 'per:date_of_death', 'org:member_of', 'org:founded_by', 'org:website', 'per:cause_of_death', 'org:political/religious_affiliation', 'org:founded', 'per:city_of_death', 'org:shareholders', 'org:number_of_employees/members', 'per:date_of_birth', 'per:city_of_birth', 'per:charges', 'per:stateorprovince_of_death', 'per:religion', 'per:stateorprovince_of_birth', 'per:country_of_birth', 'org:dissolved', 'per:country_of_death']
    # labels = ['no_relation', 'per:location_of_residence', 'per:date_of_death', 'per:charges',
            #   'org:location_of_headquarters', 'per:age', 'org:top_members/employees', 'org:founded_by',
            #   'per:location_of_birth', 'per:schools_attended', 'per:family', 'org:subsidiaries', 'org:dissolved',
            #   'per:alternate_names', 'per:location_of_death', 'org:founded', 'org:number_of_employees/members',
            #   'per:title', 'org:parents', 'org:website', 'org:member_of', 'org:shareholders', 'per:religion',
            #   'org:alternate_names', 'per:employee_of', 'per:cause_of_death', 'per:origin',
            #   'org:political/religious_affiliation', 'per:date_of_birth', 'org:members']
    tsne = TSNE(n_components=2, perplexity=8, learning_rate=100, verbose=2).fit_transform(embeddings)

    # file = open('tsne.pkl', 'wb')
    # pickle.dump([tsne, embeddings, target], file)
    # file.close()

    # file2 = open('tsne.pkl', 'rb')
    # tsne, embeddings, target = pickle.load(file2)
    # file2.close()
    vals = np.linspace(0, 1, 256)
    np.random.seed(300)

    np.random.shuffle(vals)
    # print( labels)
    
    # Including No_relation
    relations_to_plot = list(range(0,42 ))
    # relations_to_plot = (6,11,39,41)
    # relations_to_plot = (8,10,30,36)
    # relations_to_plot = (2, 25, 31)

    for i in range(len(target)):
        if target[i] not in relations_to_plot:
            tsne[i, 0] = 0
            tsne[i, 1] = 0
            target[i] = 0

    cmap = plt.cm.colors.ListedColormap(plt.cm.nipy_spectral(vals))
    # cmap = get_cmap(42)

    # print(type(tsne))

    plt.scatter(tsne[:, 0], tsne[:, 1], c=cmap(target))
    pop_n = []
    for i in relations_to_plot:
        pop_n.append(mpatches.Patch(color=cmap(i), label=labels[i]))
    # handles, _ = scatter.legend_elements(prop='colors')
    # lgd = plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    lgd = plt.legend(handles=pop_n, loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.show()
    helper.ensure_dir('images/test/', verbose=True)
    plt.savefig('images/test/' + str(global_step) + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    # #####excluding no_relations#####

    relations_to_plot = list(range(1,42 ))

    for i in range(len(target)):
        if target[i] not in relations_to_plot:
            tsne[i, 0] = 0
            tsne[i, 1] = 0
            target[i] = 0

    cmap = plt.cm.colors.ListedColormap(plt.cm.nipy_spectral(vals))
    # cmap = get_cmap(42)

    # print(type(tsne))

    plt.scatter(tsne[:, 0], tsne[:, 1], c=cmap(target))
    pop_n = []
    for i in relations_to_plot:
        pop_n.append(mpatches.Patch(color=cmap(i), label=labels[i]))
    # handles, _ = scatter.legend_elements(prop='colors')
    # lgd = plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    lgd = plt.legend(handles=pop_n, loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.show()
    helper.ensure_dir('images/test/', verbose=True)
    plt.savefig('images/test/' + str(global_step) + 'filtered.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()
