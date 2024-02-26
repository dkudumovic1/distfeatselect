# distfeatselect
distfeatselect: distributed feature selection

## Authors and Contributors
[Dzeneta Kudumovic](https://github.com/dkudumovic1), [Aida Brankovic](https://github.com/aibrank)

## Overview
-------------------

The DFS (Distributed Feature Selection) algorithm is implemented as a scikit-learn-compatible transformer, adhering to the scikit-learn API standards. This enables seamless integration into scikit-learn workflows, empowering users to incorporate distributed feature selection into their machine learning pipelines.

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

## References
The algorithms integrated into these packages have their foundations in rigorous academic research. Specifically, the methodology employed is derived from the research papers [1] and [2]. By implementing the insights and techniques outlined in this scholarly work, the packages aim to provide users with efficient and effective solutions informed by the latest advancements in the field.

[1] Brankovic, A., Falsone, A., Prandini, M., Piroddi, L. (2018). [A feature selection and classification algorithm based on randomized extraction of model populations](https://ieeexplore.ieee.org/document/7890437)

[2] Brankovic, A., Piroddi, L. (2019). [A distributed feature selection scheme with partial information sharing](https://link.springer.com/article/10.1007/s10994-019-05809-y)

## License
This project is licensed under the [MIT License](LICENSE).
