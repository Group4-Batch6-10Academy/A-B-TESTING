import seaborn as sns
from matplotlib import pyplot

class Visualization:
    """A class for plotting
    """
    def catplot(self, x, data):
        """Plotting a catagory graph
        Attributes:
            x: The x element
            data: The dataframe
        """
        ax = sns.catplot(x=x, kind="count",palette="ch:.25", data=data)
 

    def barplot(self, data,title, x, y):
        """Plot a bar graph
            Attrs:
                data: Dataframe
                title: Title of the plot
                x: xlabel
                y: ylabel
        """
        pyplot.figure(figsize=(12, 7))
        data.plot(kind="bar")
        pyplot.title(title)
        pyplot.xlabel(x)
        pyplot.ylabel(y)
        pyplot.show()