from MaCh3PythonUtils.file_handling.chain_handler import ChainHandler
import matplotlib.pyplot as plt

class ChainDiagnostics:
    def __init__(self, config_reader: ChainHandler) -> None:
        self._chain_handler = config_reader
        
    def _extract_chain_information(self, parameter_name: str | int):
        if isinstance(parameter_name, str):
            parameter_id = self._chain_handler.ttree_array.columns.get_loc(parameter_name)
        if isinstance(parameter_name, int):
            parameter_id = parameter_name
            parameter_name = self._chain_handler.ttree_array.columns[parameter_id]
        
        return self._chain_handler.ttree_array.iloc[:,parameter_id], parameter_name
        
    def __make_plot(self, fig, axs):
        if fig is None:
            fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        elif axs is None:
            axs = fig.add_subplot(1, 1, 1)
        
        return fig, axs
        
    def make_trace_plot(self, parameter_name: str | int, axs=None, fig =None):
        fig, axs = self.__make_plot(fig, axs)

        chain, parameter_name = self._extract_chain_information(parameter_name)
        axs.plot(chain, linewidth=0.5, color='darkorange')

        return fig, axs
    
    def make_autocorr_plot(self, parameter_name: str | int, axs=None, fig =None):
        fig, axs = self.__make_plot(fig, axs)

        if fig is None:
            fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        elif axs is None:
            axs = fig.add_subplot(1, 1, 1)

        chain, parameter_name = self._extract_chain_information(parameter_name)
        axs.acorr(chain, maxlags=1000, linewidth=0.5, color='darkorange')

        return fig, axs
    
    def make_posterior_hist_plot(self, parameter_name: str | int, axs=None, fig =None, is_horizontal=False):
        fig, axs = self.__make_plot(fig, axs)

        if fig is None:
            fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        elif axs is None:
            axs = fig.add_subplot(1, 1, 1)
        
        orientation = 'vertical'

        if is_horizontal:
            orientation = 'horizontal'
        
        chain, parameter_name = self._extract_chain_information(parameter_name)    
        axs.hist(chain, bins=50, density=True, linewidth=0.5, color='darkorange', alpha=0.5, orientation=orientation)

        return fig, axs
    
    def __call__(self, parameter_name: str):
        fig, axs = plt.subplots(2, 2, figsize=(15, 5))
        axs[1][1].remove()
        axs[1][0].remove()

        fig, axs[0][0] = self.make_trace_plot(parameter_name, axs=axs[0][0], fig=fig)
        fig, axs[0][1] = self.make_posterior_hist_plot(parameter_name, axs=axs[0][1], fig=fig, is_horizontal=True)

        # To share the same axis etc,
        plt.setp(axs[0][1].get_yticklabels(), visible=False)
        fig.subplots_adjust(wspace=.0)


        # fig, axs[1][0] = self.make_autocorr_plot(parameter_name, axs=axs[1][0], fig=fig)
        return fig, axs