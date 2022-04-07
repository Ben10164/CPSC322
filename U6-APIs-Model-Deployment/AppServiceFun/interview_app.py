# we are going to use the flack micro web framework
# for our web app (running our API service)

from flask import Flask, jsonify, request
import pickle
import os

# reate out web app
# flask runs by deafuly on port 5000
app = Flask(__name__)

# we are now ready to setup our first "route"
# a rout is a function that handles a request


# hey app, run the function beneath me whenever there is a get request to the root url
@app.route("/", methods=["GET"])
def index():
    # we can return content and status code
    # returns the content (the html) and the status code (200 means ok)
    return "<h1>Welcome to my app!!</h1>", 200

# now for the predict endpoint


def predict_interviewed_well(instance):
    # we need a ML model to make a prediciton for this instance
    # typically the model is trained "offline"
    # and used later "online" (e.g. via this web app)
    # enter pickling

    # now that we have made tree.p, we can unpickle it!
    # unpickle tree.p into header and the tree
    infile = open("tree.p", "rb")  # r for read, b for binary
    header, tree = pickle.load(infile)
    infile.close()
    print(header)
    print(tree)
    print(instance)

    try:
        prediction = tdidt_predict(header, tree, instance)
        return prediction
    except:
        print("error")
        return None
    pass


def tdidt_predict(header, tree, instance):
    # recursively traverse the tree
    # we need to know wher we are in the tree
    # are we at a leaf node? (base case)
    # if not, we are at an attribute node
    #
    info_type = tree[0]
    if info_type == "Leaf":
        return tree[1]
    # we dont have a base case, so we need to recurse
    # we need to amtch the attribute's value in the instance
    # with the appropriate value list
    # a for loop that traverses through each
    # value list
    # recurse on match with instance value
    att_index = header.index(tree[1])
    print("instance", instance[att_index])

    for i in range(2, len(tree)):
        value_list = tree[i]
        # checking if the value of the instance is in the value list
        print("value list", value_list)
        if value_list[1] == instance[att_index]:
            return tdidt_predict(header, value_list[2], instance)
    # if info_type == "Attribute":
    #     attr_name = tree[1]
    #     for attr_branch in tree[2:]:
    #         if attr_branch[1] in instance:
    #             print(attr_branch[1])
    #             return tdidt_predict(header, attr_branch, instance)
    # TODO: finish this


@app.route("/predict", methods=["GET"])
def predict():
    # parse the query string to get out
    # instance attribute values from the clietn
    level = request.args.get("level", "")
    lang = request.args.get("lang", "")
    tweets = request.args.get("tweets", "")
    phd = request.args.get("phd", "")
    print("level:", level, "lang:", lang, "tweets:", tweets, "phd:", phd)

    # TODO: fix the hardcoding
    prediction = predict_interviewed_well([level, lang, tweets, phd])
    # if anything goes wrong in the function above, it will return none
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    return "Error making prediction", 400  # error code 400 is bad request


if __name__ == "__main__":
    # TODO: turn debug off when deploy to "productions"
    # we are going to use heroku (PaaS platofrom as a service)
    # thera are a fewawys to deploy a flask to heroku
    # we will use 2.B.
    port = os.environ.get("PORT", 5001)
    app.run(debug=False, port=port, host="0.0.0.0")
