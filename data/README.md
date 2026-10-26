# Dataset

`fake_reviews.csv` is the **Fake Reviews Dataset** by Salminen et al. (2022), included unmodified.
`train_model.py` uses it to train the model in `models/`.

- **Source:** <https://osf.io/tyue9/>
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Contents:** 40,432 Amazon reviews across 10 product categories. Half are original
  reviews (`label = OR`) and half are computer-generated (`label = CG`).
- **Columns:** `category`, `rating`, `label`, `text_`

**Citation:** Salminen, J., Kandpal, C., Kamel, A. M., Jung, S., & Jansen, B. J. (2022).
*Creating and detecting fake reviews of online products.* Journal of Retailing and
Consumer Services, 64, 102771.
