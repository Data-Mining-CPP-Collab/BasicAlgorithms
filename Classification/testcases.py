import sys
import classification
import pandas
import numpy
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

# Naive Bayes using sklearn 
class NaiveBayes:
    def fit(self, x,y):
        # NB needs the categories to be converted to numbers 
        # The ordinal encoder does exactly that 
        self.encoder = OrdinalEncoder()
        self.NB = CategoricalNB()
        # encode the data 
        xenc = self.encoder.fit_transform(x)
        # fit the naive bayes model 
        self.NB.fit(xenc,y)
    def predict(self, x):
        # to predict, we first need to encode the data (using the same encoder)
        # and then predict using the NB model 
        return self.NB.predict(self.encoder.transform(x))
    def to_dict(self):
        # Just to have some representation for the classifier
        return {"encoder": str(self.encoder), "nb": str(self.NB)}
        
CATMAP = {"low": 0, "medium": 1, "high": 2}
BINMAP = {False: 0, True: 1}
COLORS = {0: "Red", 1: "Blue"}

    
def plot(title, xs, ys, predys, mapping=CATMAP):
    markers = dict(zip(set(ys), ["+", "_", "*", "^", "s", "D"]))
    x1s = {}
    x2s = {}
    colors = {}
    for m in markers:
        x1s[m] = []
        x2s[m] = []
        colors[m] = []
    
    for x,y,py in zip(xs,ys,predys):
        x1s[y].append(mapping[x[0]]+random.gauss(0,0.05))
        # If we only have one feature, use random noise for the y-axis
        if len(x) == 1:
            x2s[y].append(random.gauss(0,0.2))  # Larger spread for better visualization
        else:
            x2s[y].append(mapping[x[1]]+random.gauss(0,0.05))
        colors[y].append(COLORS[py])
    
    for m in markers:
        plt.scatter(x1s[m], x2s[m], c=colors[m], marker=markers[m])
    
    # Add labels
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2" if len(xs[0]) > 1 else "Random jitter")
    plt.title(title)
    plt.show()

def evaluate(prefix, y, predy):
    correct = 0
    for p,v in zip(predy, y):
        if p == v: correct += 1
    print("%sAccuracy: %.2f"%(prefix, correct*1.0/len(y)))
    
def get_columns(rows, columns, single=False):
    if single:
        return [row[columns[0]] for row in rows]
    return [[row[c] for c in columns] for row in rows]
    


MODELS = {"Decision Tree": classification.DecisionTree, "Naive Bayes": NaiveBayes}
    
CLASSIFICATION_TESTS = [
    "Predict class from two categories (1+2)", 
    "Predict class from two categories (3+4)", 
    "Predict class from all four categories", 
    "Predict class from three categories (2-4)", 
    "Predict class from two binary attributes (1+2)", 
    "Predict class from three binary attributes (3+4)", 
    "Predict class from wrong categorical attributes", 
    "Predict class from wrong binary attributes",
    "Predict class from mixed attributes (2 cat + 2 bin)",
    "Predict class using single categorical attribute",
    "Predict class using single binary attribute",
    "Predict class from all attributes",
    "Predict class from two categories (3+4) with pruning, max depth was now one"  # New test case
]

# Update the classification_testcase function by adding the new case:
def classification_testcase(training, validation, n, visualize=True, model="Decision Tree"):
    print("running test:", CLASSIFICATION_TESTS[n])
    if n == 0:
        columns = ["cat1", "cat2"]
        target = ["cls3"]
        mapping = CATMAP
    elif n == 1:
        columns = ["cat3", "cat4"]
        target = ["cls3"]
        mapping = CATMAP
    # ... (keep existing test cases)
    elif n == 11:
        columns = ["cat1", "cat2", "cat3", "cat4", "bin1", "bin2", "bin3", "bin4", "bin5"]
        target = ["cls3"]
        visualize = False
    elif n == 12:  # New pruned test case
        # we just do everything here and return 
        columns = ["cat3", "cat4"]
        target = ["cls3"]
        mapping = CATMAP
        m = classification.DecisionTree(max_depth=1)  # Create tree with pruning
        tx = get_columns(training, columns)
        ty = get_columns(training, target, single=True)
        m.fit(tx, ty)
        print(json.dumps(m.to_dict(), indent=4))
        predty = m.predict(tx)
        evaluate(model + " training ", ty, predty)
        vx = get_columns(validation, columns)
        vy = get_columns(validation, target, single=True)
        predvy = m.predict(vx)
        evaluate(model + " validation ", vy, predvy)
        
        if visualize:
            plot(model + " training set (pruned), depth is only 1", tx, ty, predty, mapping)
            plot(model + " validation set (pruned), depth is only 1", vx, vy, predvy, mapping)
        return  # Return early since we handled everything in this branch
        
    m = MODELS[model]()
    tx = get_columns(training, columns)
    ty = get_columns(training, target, single=True)
    m.fit(tx, ty)
    print(json.dumps(m.to_dict(), indent=4))
    predty = m.predict(tx)
    evaluate(model + " training ", ty, predty)
    vx = get_columns(validation, columns)
    vy = get_columns(validation, target, single=True)
    predvy = m.predict(vx)
    evaluate(model + " validation ", vy, predvy)
    
    if visualize:
        plot(model + " training set", tx, ty, predty, mapping)
        plot(model + " validation set", vx, vy, predvy, mapping)

def main(auto=False, nb=False, steps=[]):
    random.seed(1337)
    df = pandas.read_csv("testdata.csv")
    model = "Decision Tree"
    if nb:
        model = "Naive Bayes"
    training = []
    validation = []
    for i,row in df.iterrows():
        if random.random() > 0.85:
            validation.append(row)
        else:
            training.append(row)
    
    if auto:
        for i,t in enumerate(CLASSIFICATION_TESTS):
            print("-"*80)
            classification_testcase(training, validation, i, False, model)
        return
    tests = CLASSIFICATION_TESTS
    while True:
        print("Which test case do you want to run?")
        for i,t in enumerate(tests):
            print(f"   {i} {t}")
        print("   q exit")
        if steps:
            x = steps[0]
            del steps[0]
        else:
            x = input("> ")
        if x in [str(i) for i,_ in enumerate(tests)]:
            classification_testcase(training, validation, int(x), model=model)
        elif x == "q":
            print("Bye")
            sys.exit(0)
        else:
            print("Please select a test case, r or q")
        print()

if __name__ == "__main__":
    if "--help" in sys.argv:
        print("Usage: testcases.py [--auto] steps")
        print("   --auto runs the tests automatically")
        print("   --naive-bayes uses the Naive Bayes classifier from sklearn; useful as a comparison")
        print("   <steps> is a sequence of inputs that are passed to the menu before it accepts manual input.")
        print("           This allows you to run e.g. 'python testcases.py 12q' to run test cases 1 and 2")
        print("           in sequence, followed by q(uit)")
        print("           Essentially, this allows you to repeatedly run any test/combination of tests without")
        print("           having to navigate the menu every time.")
        sys.exit(0)
    else:
        main("--auto" in sys.argv, "--bayes" in sys.argv or "--naive-bayes" in sys.argv or "-n" in sys.argv,
             list("".join([arg for arg in sys.argv[1:] if "-" not in arg])))