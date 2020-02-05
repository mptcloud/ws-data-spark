Scaling function
----------------

In our `pandas_solution_modeling.ipynb` notebook, we start by creating a basic linear scaling function
that takes the POI scores (whatever they may be) and maps them to the [-10, 10] range.

We note that this isn't the only way to perform this scaling. Depending on the kind of sensitivity being
looked for in a certain region, we can easily replace this scaling function with a tanh (hyperbolic tangent)
function, for example.


How do we model the POI's?
--------------------------

The POI's can be modeled/viewed in two ways:
1. Using spatial analysis
2. Using graph networks


Outlier removal
---------------

As some of the plots within the plots folder show, both POI1 and POI4 have outliers, with some requests lying well over the
10,000 km mark.


Density measure -- a basic spatial approach
-------------------------------------

Once these outliers are removed, and we use the density calculated earlier in the analysis as the scores to be scaled, we end up with the
following distribution of scores.

**Ranking scores** -- POI1: 4.5, POI3: 10, POI4: -10


Graph networks
--------------

One way of modeling each POI's influence/popularity is via a star S_k graph network (Here, k refers to the number of
request IDs associated to a POI). So, POI1 forms a S<sub>8749</sub> graph network, while POI4, on the other hand, forms a
S<sub>422</sub> graph network. In this case, each POI serves as the internal node with all of the associated requests serving
as the leaves of the network.


Degree centrality measure
-------------------------

In this scenario, going purely by the degree centrality of the central/internal node, we can rank POI's based on the
number of connections they have and scale that to [-10, 10].

**Ranking scores** -- POI1: 9.874, POI3: 10, POI4: -10, which isn't a very good spread of scores.


An important point regarding the analysis
-----------------------------------------

The final scores that we get are a result of the scores that we use as inputs for the scaling function. In this scenario, then,
we need additional information to determine what aspect of the POI's are we emphasizing.

For example, using the density measure as our base scores means that we are valuing density as the prime measure for a POI. In other words,
POIs for which there exists a denser cluster of requests will naturally get higher scores. This varies drastically from the degree centrality
approach, where distance does not play a role.

This question of what to value for each POI is a decision to be made by an analyst with domain expertise. Valuing things differently will
lead to different hierarchy of ranking scores for each POI.


Other approaches (weighted linkage, used in `pandas_solution_modeling_2.ipynb`)
-------------------------------------------------------------------------------

Within the graph network theory approach, we can assign a weight to each edge equaling the distance between the POI and the request ID raised to a
certain power. This is what is done by the linkage_weighted_mean method in `pandas_solution_modeling_2.ipynb`.

We can then look at how the ranking scores of each POI shift as we change the exponent of the distance in the weighted mean.

**With an exponent of -3**:
POI1: 10, POI3: -8, POI4: -10

**With an exponent of 1**:
POI1: -5, POI3: 10, POI4: -10

**With an exponent of 9**:
POI1: 10, POI3: 1.4, POI4: -10

As the results above show, changing the exponent creates a distinct difference in the rankings. When the exponent is negative, 
requests that are closer to the POI are valued more and hence, POI1 is ranked first. With an increasingly positive exponent, 
requests that are farther away from the POI are valued more, which, in turn, shows how POI3 gives way to POI1 owing to the 
greater number of requests away from the POI. In this regard, the loose clustering of POI1 will help it achieve a higher 
rank than POI3 with either a negative or extremely large positive exponent.

This analysis only serves to show that some kind of domain expertise is needed to come up with a ranking that reflects the 
reality of the situation. Without any additional information, it isn't possible to come up with a ranking that will accurately 
model the situation in question, but what we do have are candidate models worth further inspection.



