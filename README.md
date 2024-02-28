# distfeatselect
distfeatselect: distributed feature selection

## Authors and Contributors
[Dzeneta Kudumovic](https://github.com/dkudumovic1), [Aida Brankovic](https://github.com/aibrank)

## Overview
distfeatselect is a Python package implementing an innovative Distributed Feature Selection algorithm based on vertical data partitioning and distributed searching. By dividing features into subsets and assigning dedicated processors for local searches, the algorithm achieves parallelism and scalability, making it suitable for large-scale datasets. The distributed architecture enhances efficiency by operating on smaller search spaces, reducing computational time. Moreover, algorithm's tendency to produce simple model structures enhances interpretability and robustness. 

The DFS (Distributed Feature Selection) algorithm is implemented as a scikit-learn-compatible transformer, adhering to the scikit-learn API standards. This enables seamless integration into scikit-learn workflows, empowering users to incorporate distributed feature selection into their machine learning pipelines.

To get started with distfeatselect, explore our provided notebooks, which offer hands-on examples and demonstrations of the package's functionality.

### Requirements
The package dcor depends on the following libraries:
- numpy
- pandas,
- statsmodels
- scikit-learn
- dcor

## Installation
distfeatselect is on PyPi and can be installed using pip:
```bash
pip install distfeatselect
```

## Documentation
- [Quick start notebook](https://dkudumovic1.github.io/distfeatselect/distfeatselect_Quick_start.ipynb)
- [DFS API Docs](https://dkudumovic1.github.io/distfeatselect/dfs.html)
- [RFS API Docs](https://dkudumovic1.github.io/distfeatselect/rfs.html)
  
## References
The algorithms integrated into these packages have their foundations in rigorous academic research. Specifically, the methodology employed is derived from the research papers [1] and [2]. By implementing the insights and techniques outlined in this scholarly work, the packages aim to provide users with efficient and effective solutions informed by the latest advancements in the field.

[1] Brankovic, A., Falsone, A., Prandini, M., Piroddi, L. (2018). [A feature selection and classification algorithm based on randomized extraction of model populations](https://ieeexplore.ieee.org/document/7890437)

[2] Brankovic, A., Piroddi, L. (2019). [A distributed feature selection scheme with partial information sharing](https://link.springer.com/article/10.1007/s10994-019-05809-y)

## License
This project is licensed under the [MIT License](LICENSE).
