# 🏒 IFT-6758 Project
[Repo URL](https://github.com/AxelBogos/IFT6758-Project)

Description

- [👩‍🔬👨‍🔬 Authors](#-authors)
- [👀️ Demo](#️-demo)
- [📐 Environment Variables](#-environment-variables-)
- [⚙️ Run Locally](#️-run-locally)
- [✒️ Documentation](#️-documentation)
- [☑️ Milestones](#️-milestones)
- [📡 Deployment](#-deployment)
- [🔨 Appendix](#-appendix)

## 👩‍🔬👨‍🔬 Authors[](https://)

- [@Axel Bogos](https://www.github.com/AxelBogos)
- [@Marc-André Gagné](https://www.github.com/MAGjagger)
- [@Van Binh Truong](https://www.github.com/VanBinhTruong)
- [@Houda Saadaoui](https://www.github.com/houdasaad)


## 👀️ Demo
-> See interactive plot in `figures/Q6_plot.html` <br>
-> Blog post repo [here!](https://github.com/MAGjagger/ift6758-Blog)

## 📐 Environment Variables
- `COMET_API_KEY` : Your personal API key to use *comet.ml*. Create a free acount and see `Account -> Settings -> Developer Information`

## ⚙️ Run Locally

1. Clone the project

```bash
  git clone https://github.com/AxelBogos/IFT6758-Project.git
```

2. Go to the project directory

```bash
  cd project
```

3. Install virtual environment

```bash
conda env create --name YOUR_ENV_NAME --file=environment.yml
conda activate YOUR_ENV_NAME   
```
4. Get the data (~3.15Gb)

```python
python ift6758/data/data-acquisition.py
```

5. Run tidy data script (~5-10 min depending on your I/O speed)

```python
python ift6758/features/tidy_dataframe_builder.py
```

6. Run Plotting Functions

```python
python ift6758/visualizations/plots.py
```

7. Have fun exploring any notebook and figures in 
`./notebooks` and `./figures`

## ✒️ Documentation

[Documentation](https://linktodocumentation) or just write more details here

## ☑️ Milestones

- [x] Milestone 1 - October 15 2021, 11:59PM
- [ ] Milestone 2 - November 26 2021, 11:59PM
- [ ] Milestone 3 - TBA

## 📡 Deployment

1. Blog post repo [here!](https://github.com/MAGjagger/ift6758-Blog)

2. Clone the project

```bash
  git clone https://github.com/MAGjagger/ift6758-Blog.git
```
3. Go to the project directory

```bash
  cd project
```

4. Execute the server. See [here](https://github.com/MAGjagger/ift6758-Blog/blob/main/README.md) for more details about the jekyll installation. 
```bash
  bundle exec jekyll serve
```

## 🔨 Appendix

TBA
