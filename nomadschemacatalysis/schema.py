import numpy as np
import os

from ase.data import chemical_symbols

from nomad.metainfo import (
    Quantity,
    Section,
    SubSection,
    Package)

from nomad.units import ureg

from nomad.datamodel.metainfo.eln import (
    #CompositeSystem,
    Measurement)

from nomad.datamodel.metainfo.basesections import CompositeSystem, System

from nomad.datamodel.data import ArchiveSection

from nomad.datamodel.results import (Results, Material, Properties, CatalyticProperties,
                                     CatalystCharacterization, CatalystSynthesis, Reactivity)
from nomad.datamodel.data import EntryData, UseCaseElnCategory

from .catalytic_measurement import (
    CatalyticReactionData, CatalyticReactionData_core, Feed, Reagent, Conversion, Rates, Reactor_setup, ReactionConditions,
    add_activity
    )

from nomad.datamodel.results import Product, Reactant

from nomad.datamodel.metainfo.plot import PlotSection, PlotlyFigure
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json

from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
)

m_package = Package(name='catalysis')


def add_catalyst(archive):
    '''Adds metainfo structure for catalysis data.'''
    if not archive.results:
        archive.results = Results()
    if not archive.results.properties:
        archive.results.properties = Properties()
    if not archive.results.properties.catalytic:
        archive.results.properties.catalytic = CatalyticProperties()
    if not archive.results.properties.catalytic.catalyst_characterization:
        archive.results.properties.catalytic.catalyst_characterization = CatalystCharacterization()
    if not archive.results.properties.catalytic.catalyst_synthesis:
        archive.results.properties.catalytic.catalyst_synthesis = CatalystSynthesis()


class Preparation(ArchiveSection):
    # m_def = Section(label_quantity=)

    preparation_method = Quantity(
        type=str,
        shape=[],
        description="""
          classification of dominant preparation step
          """,
        a_eln=dict(
            component='EnumEditQuantity', props=dict(
                suggestions=['precipitation', 'hydrothermal', 'flame spray pyrolysis',
                             'impregnation', 'calcination', 'unknown']))
    )

    preparator = Quantity(
        type=str,
        shape=[],
        description="""
        The person or persons preparing the sample in the lab.
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(
            suggestions=['A. Trunschke',
                         'R. Schlögl'])),
        #repeats=True
    )

    preparing_institution = Quantity(
        type=str,
        shape=[],
        description="""
        institution at which the sample was prepared
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(
            suggestions=['Fritz-Haber-Institut Berlin / Abteilung AC',
                         'Fritz-Haber-Institut Berlin / ISC']))
    )

    def normalize(self, archive, logger):
        super(Preparation, self).normalize(archive, logger)

        add_catalyst(archive)

        if self.preparation_method is not None:
            archive.results.properties.catalytic.catalyst_characterization.preparation_method = self.preparation_method


class SurfaceArea(ArchiveSection):
    m_def = Section(label_quantity='method_surface_area_determination')

    surface_area = Quantity(
        type=np.float64,
        unit=("m**2/g"),
        a_eln=dict(
            component='NumberEditQuantity', defaultDisplayUnit='m**2/g')
    )

    method_surface_area_determination = Quantity(
        type=str,
        shape=[],
        description="""
          description of method to measure surface area
          """,
        a_eln=dict(
            component='EnumEditQuantity', props=dict(
                suggestions=['BET', 'H2-TPD', 'N2O-RFC',
                             'Fourier Transform Infrared Spectroscopy (FTIR) of adsorbates',
                             'unknown']))
    )


    def normalize(self, archive, logger):
        super(SurfaceArea, self).normalize(archive, logger)

        add_catalyst(archive)

        # if self.method_surface_area is not None:
        archive.results.properties.catalytic.catalyst_characterization.surface_area = self.surface_area
        archive.results.properties.catalytic.catalyst_characterization.method_surface_area = self.method_surface_area_determination


class CatalystSample(CompositeSystem, EntryData):
    """
    This schema is originally adapted to map the data of the clean Oxidation dataset (JACS,
    https://doi.org/10.1021/jacs.2c11117) The descriptions in the quantities
    represent the instructions given to the user who manually curated the data.
    """

    m_def = Section(
        label='Heterogeneous Catalysis - Catalyst Sample',
        #a_eln=dict(hide=['cas_uri', 'cas_number', 'cas_name', 'inchi', 'inchi_key',
        #                 'smile', 'canonical_smile', 'cas_synonyms', 'molecular mass']),
        categories=[UseCaseElnCategory],
    )

    preparation_details = SubSection(section_def=Preparation)

    surface = SubSection(section_def=SurfaceArea)

    storing_institution = Quantity(
        type=str,
        shape=[],
        description="""
        institution at which the sample is stored
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(
            suggestions=['Fritz-Haber-Institut Berlin / Abteilung AC',
                         'Fritz-Haber-Institut Berlin / ISC', 'TU Berlin / BasCat']))
    )

    catalyst_type = Quantity(
        type=str,
        shape=[],
        description="""
          classification of catalyst type
          """,
        a_eln=dict(
            component='EnumEditQuantity', props=dict(
                suggestions=['bulk catalyst', 'supported catalyst', 'single crystal',
                             '2D catalyst', 'other', 'unkown'])),
    )

    form = Quantity(
        type=str,
        shape=[],
        description="""
          classification of physical form of catalyst
          """,
        a_eln=dict(component='EnumEditQuantity', props=dict(
            suggestions=['sieve fraction', 'powder', 'thin film']))
    )

    def normalize(self, archive, logger):
        super(CatalystSample, self).normalize(archive, logger)

        add_catalyst(archive)

        if self.catalyst_type is not None:
            archive.results.properties.catalytic.catalyst_synthesis.catalyst_type = self.catalyst_type
        # if self.surface_area is not None:
        #     archive.results.properties.catalytic.catalyst_characterization.surface_area = self.surface_area.surfacearea
        #     archive.results.properties.catalytic.catalyst_characterization.method_surface_area = self.surface_area.method_surface_area_determination
        if self.preparation_details is not None:
            archive.results.properties.catalytic.catalyst_synthesis.preparation_method = self.preparation_details.preparation_method

class CatalyticReaction_core(ArchiveSection):
    sample_reference = Quantity(
        type=CatalystSample,
        description="""
        link to a catalyst sample entry
        """,
        a_eln=dict(component='ReferenceEditQuantity')
    )

    reaction_class = Quantity(
        type=str,
        description="""
        highlevel classification of reaction
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(suggestions=[
            'Oxidation', 'Hydrogenation', 'Dehydrogenation', 'Cracking', 'Isomerisation', 'Coupling']
        )))

    reaction_name = Quantity(
        type=str,
        description="""
          name of reaction
          """,
        a_eln=dict(
            component='EnumEditQuantity', props=dict(suggestions=[
                'Alkane Oxidation', 'Oxidation of Ethane', 'Oxidation of Propane',
                'Oxidation of Butane', 'CO hydrogenation', 'Methanol Synthesis', 'Fischer-Tropsch',
                'Water gas shift reaction', 'Ammonia Synthesis', 'Ammonia decomposition'])))

    experiment_handbook = Quantity(
        description="""
        was the experiment performed according to a handbook
        """,
        type=str,
        shape=[],
        a_eln=dict(component='FileEditQuantity')
    )

    institute = Quantity(
        type=str,
        shape=[],
        description="""
        institution at which the measurement was performed
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(
            suggestions=['Fritz-Haber-Institut Berlin / Abteilung AC',
                         'Fritz-Haber-Institut Berlin / ISC',
                         'TU Berlin, BASCat', 'HZB', 'CATLAB']))
    )

    experimenter = Quantity(
        type=str,
        shape=[],
        description="""
        person that performed or started the measurement
        """,
        a_eln=dict(component='EnumEditQuantity')
    )

# def make_plot(a,b):
    
#     return()

class SimpleCatalyticReaction(Measurement, EntryData):
    reaction_condition = SubSection(section_def=ReactionConditions, a_eln=ELNAnnotation(label='Reaction Conditions'))

class CatalyticReaction(CatalyticReaction_core, PlotSection, EntryData):

    m_def = Section(
        label='Heterogeneous Catalysis - Activity Test',
        categories=[UseCaseElnCategory]
    )

    data_file = Quantity(
        type=str,
        description="""
        excel or csv file that contains results of a catalytic measurement with
        temperature, (pressure,) gas feed composition, yield, rates and selectivities
        """,
        a_eln=dict(component='FileEditQuantity'),
        a_browser=dict(adaptor='RawFileAdaptor'))

    reactor_setup = SubSection(section_def=Reactor_setup, a_eln=ELNAnnotation(label='Reactor Setup'))

    reaction_conditions = SubSection(section_def=Feed, a_eln=ELNAnnotation(label='Reaction Conditions'))
    reaction_results = SubSection(section_def=CatalyticReactionData, a_eln=ELNAnnotation(label='Reaction Results'))

    measurement_details = SubSection(section_def=Measurement)

    def normalize(self, archive, logger):

        if (self.data_file is None):  # and (self.data_file_h5 is None):
            return

        if ((self.data_file is not None) and (os.path.splitext(
                self.data_file)[-1] != ".csv" and os.path.splitext(
                self.data_file)[-1] != ".xlsx")):
            raise ValueError("Unsupported file format. Only xlsx and .csv files")

        if self.data_file.endswith(".csv"):
            with archive.m_context.raw_file(self.data_file) as f:
                import pandas as pd
                data = pd.read_csv(f.name).dropna(axis=1, how='all')
        elif self.data_file.endswith(".xlsx"):
            with archive.m_context.raw_file(self.data_file) as f:
                import pandas as pd
                data = pd.read_excel(f.name, sheet_name=0)

        data.dropna(axis=1, how='all', inplace=True)
        feed = Feed()
        cat_data = CatalyticReactionData()
        reagents = []
        reagent_names = []
        # reactants = []
        # reactant_names = []
        products = []
        product_names = []
        conversions = []
        conversions2 = []
        conversion_names = []
        conversion_list = []
        rates = []
        number_of_runs = 0
        for col in data.columns:

            if len(data[col]) < 1:
                continue
            col_split = col.split(" ")
            if len(col_split) < 2:
                continue

            if len(data[col]) > number_of_runs:
                number_of_runs = len(data[col])

            if col_split[0] == "x":
                reagent = Reagent(name=col_split[1], gas_concentration_in=data[col])
                reagent_names.append(col_split[1])
                reagents.append(reagent)
            if col_split[0] == "mass":
                catalyst_mass_vector = data[col]
                feed.catalyst_mass = catalyst_mass_vector[0]

            if col_split[0] == "temperature":
                if "K" in col_split[1]:
                    cat_data.temperature = np.nan_to_num(data[col])
                else:
                    cat_data.temperature = np.nan_to_num(data[col])*ureg.celsius

            if col_split[0] == "TOS":
                cat_data.time_on_stream = data[col]

            if col_split[0] == "C-balance":
                cat_data.c_balance = np.nan_to_num(data[col])

            if col_split[0] == "GHSV":
                feed.gas_hourly_space_velocity = np.nan_to_num(data[col])

            if col_split[0] == "Vflow":
                feed.set_total_flow_rates = np.nan_to_num(data[col])

            if col_split[0] == "pressure":
                cat_data.pressure = np.nan_to_num(data[col])

            if col_split[0] == "r":  # reaction rate
                rate = Rates(name=col_split[1], reaction_rate=np.nan_to_num(data[col]))
                # if col_split[1] in reagent_names:
                #     reactant.reaction_rate = data[col]
                # rate.reaction_rate = data[col]
                rates.append(rate)

            if len(col_split) < 3 or col_split[2] != '(%)':
                continue

            if col_split[0] == "x_p":  # conversion, based on product detection
                conversion = Conversion(name=col_split[1], conversion=np.nan_to_num(data[col]),
                                        type='product-based conversion', conversion_product_based=np.nan_to_num(data[col]))
                for i, p in enumerate(conversions):
                    if p.name == col_split[1]:
                        conversion = conversions.pop(i)

                conversion.conversion_product_based = np.nan_to_num(data[col])

                conversion_names.append(col_split[1])
                conversion_list.append(data[col])
                conversions.append(conversion)

            if col_split[0] == "x_r":  # conversion, based on reactant detection
                #if data['x '+col_split[1]+' (%)'] is not None:
                try:
                    conversion2 = Reactant(name=col_split[1], conversion=np.nan_to_num(data[col]), gas_concentration_in=(np.nan_to_num(data['x '+col_split[1]+' (%)']))/100)
                    conversions2.append(conversion2)
                except KeyError:
                    pass
                try:
                    conversion2 = Reactant(name=col_split[1], conversion=np.nan_to_num(data[col]), gas_concentration_in=np.nan_to_num(data['x '+col_split[1]]))
                    conversions2.append(conversion2)
                except KeyError:
                    pass
                finally:    
                    conversion = Conversion(name=col_split[1], conversion=np.nan_to_num(data[col]), type='reactant-based conversion', conversion_reactant_based = np.nan_to_num(data[col]))
                for i, p in enumerate(conversions):
                    if p.name == col_split[1]:
                        conversion = conversions.pop(i)
                conversion.conversion_reactant_based = data[col]
                conversions.append(conversion)

            if col_split[0] == "S_p":  # selectivity
                product = Product(name=col_split[1], selectivity=np.nan_to_num(data[col]))
                # for i, p in enumerate(rates):
                #     if p.name == col_split[1]:
                #         rate = rates.pop(i)
                #         product.reaction_rate=rate.reaction_rate
                #         break

                products.append(product)
                product_names.append(col_split[1])

        feed.reagents = reagents
        if data['step'] is not None:
            feed.runs = data['step']
            cat_data.runs = data['step']
        else:
            cat_data.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        cat_data.products = products
        cat_data.reactants_conversions = conversions
        cat_data.rates = rates
        self.reaction_conditions = feed
        self.reaction_results = cat_data

        super(CatalyticReaction, self).normalize(archive, logger)

        add_activity(archive)

        if conversions2 is not None:
            archive.results.properties.catalytic.reactivity.reactants = conversions2
        if cat_data.temperature is not None:
            archive.results.properties.catalytic.reactivity.test_temperatures = cat_data.temperature
        if cat_data.pressure is not None:
            archive.results.properties.catalytic.reactivity.pressure = cat_data.pressure
        # if feed.space_velocity is not None:
        #     archive.results.properties.catalytic.reactivity.gas_hourly_space_velocity = feed.space_velocity
        if products is not None:
            archive.results.properties.catalytic.reactivity.products = products
        if self.reaction_name is not None:
            archive.results.properties.catalytic.reactivity.reaction_name = self.reaction_name
            archive.results.properties.catalytic.reactivity.reaction_class = self.reaction_class

        if self.sample_reference is not None:
            if not archive.results.properties.catalytic.catalyst_characterization:
                archive.results.properties.catalytic.catalyst_characterization = CatalystCharacterization()
            if not archive.results.properties.catalytic.catalyst_synthesis:
                archive.results.properties.catalytic.catalyst_synthesis = CatalystSynthesis()

            if self.sample_reference.catalyst_type is not None:
                archive.results.properties.catalytic.catalyst_synthesis.catalyst_type = self.sample_reference.catalyst_type
            if self.sample_reference.preparation_details is not None:
                archive.results.properties.catalytic.catalyst_synthesis.preparation_method = self.sample_reference.preparation_details.preparation_method
            if self.sample_reference.surface is not None:
                archive.results.properties.catalytic.catalyst_characterization.surface_area = self.sample_reference.surface.surface_area

        if self.sample_reference.elemental_composition is not None:
            if not archive.results:
                archive.results = Results()
            if not archive.results.material:
                archive.results.material = Material()

            try:
                archive.results.material.elemental_composition = self.sample_reference.elemental_composition
  
            except Exception as e:
                logger.warn('Could not analyse elemental compostion.', exc_info=e)
        
        if self.reaction_results.time_on_stream is not None:
            x=self.reaction_results.time_on_stream.to('hour')
            x_text="time (h)"
        else:
            x=self.reaction_results.runs
            x_text="steps" 

        if self.reaction_results.temperature is not None:
            fig = px.line(x=x, y=self.reaction_results.temperature.to("celsius"))
            fig.update_xaxes(title_text=x_text)
            fig.update_yaxes(title_text="Temperature (°C)")
            self.figures.append(PlotlyFigure(label='figure Temperature', figure=fig.to_plotly_json()))
            self.reaction_results.figures.append(PlotlyFigure(label='Temperature', figure=fig.to_plotly_json()))

        
        fig0 = go.Figure()
        for i,c in enumerate(self.reaction_results.products):
            fig0.add_trace(go.Scatter(x=self.reaction_results.runs, y=self.reaction_results.products[i].selectivity, name=self.reaction_results.products[i].name))
        fig0.update_layout(title_text="Selectivity", showlegend=True)
        fig0.update_xaxes(title_text="measurement points")
        fig0.update_yaxes(title_text="Selectivity (%)")
        self.figures.append(PlotlyFigure(label='figure Selectivity', figure=fig0.to_plotly_json()))

        fig1 = go.Figure()
        for i,c in enumerate(self.reaction_results.reactants_conversions):
            fig1.add_trace(go.Scatter(x=x, y=self.reaction_results.reactants_conversions[i].conversion, name=self.reaction_results.reactants_conversions[i].name))
        fig1.update_layout(title_text="Conversion", showlegend=True)
        fig1.update_xaxes(title_text=x_text)
        fig1.update_yaxes(title_text="Conversion (%)")
        self.figures.append(PlotlyFigure(label='figure Conversion', figure=fig1.to_plotly_json()))

        if self.reaction_results.rates is not None:
            fig = go.Figure()
            for i,c in enumerate(self.reaction_results.rates):
                fig.add_trace(go.Scatter(x=x, y=self.reaction_results.rates[i].reaction_rate, name=self.reaction_results.rates[i].name))
            fig.update_layout(title_text="Rates", showlegend=True)
            fig.update_xaxes(title_text=x_text)
            fig.update_yaxes(title_text="reaction rates")
            self.reaction_results.figures.append(PlotlyFigure(label='Rates', figure=fig.to_plotly_json()))
            try:
                fig2 = px.line(x=self.reaction_results.temperature.to('celsius'), y=[self.reaction_results.rates[0].reaction_rate])
                fig2.update_xaxes(title_text="Temperature (°C)")
                fig2.update_yaxes(title_text="reaction rate (mmol(H2)/gcat/min)")
                self.figures.append(PlotlyFigure(label='figure rates', figure=fig2.to_plotly_json()))
            except:
                print("No rates defined")

        # if self.reaction_conditions.set_temperature is not None:
        #     fig4 = px.scatter(x=self.reaction_conditions.runs, y=self.reaction_conditions.set_temperature.to('kelvin'))
        #     fig4.update_layout(title_text="Temperature")
        #     fig4.update_xaxes(title_text="measurement points",) 
        #     fig4.update_yaxes(title_text="Temperature (K)")
        #     self.reaction_conditions.figures.append(PlotlyFigure(label='Temperature', figure=fig4.to_plotly_json()))

        # if self.reaction_conditions.reagents is not None:
        #     fig5 = go.Figure()
        #     for i,c in enumerate(self.reaction_conditions.reagents):
        #         if self.reaction_conditions.reagents[0].flow_rate is not None:
        #             fig5.add_trace(go.Scatter(x=x, y=self.reaction_conditions.reagents[i].flow_rate, name=self.reaction_conditions.reagents[i].name))
        #             y5_text="Flow rates ()"
        #         elif self.reaction_conditions.reagents[0].gas_concentration_in is not None:
        #             fig5.add_trace(go.Scatter(x=x, y=self.reaction_conditions.reagents[i].gas_concentration_in, name=self.reaction_conditions.reagents[i].name))    
        #             y5_text="gas concentrations"
        #     fig5.update_layout(title_text="Gas feed", showlegend=True)
        #     fig5.update_xaxes(title_text=x_text) 
        #     fig5.update_yaxes(title_text=y5_text)
        #     self.reaction_conditions.figures.append(PlotlyFigure(label='Feed Gas', figure=fig5.to_plotly_json()))

        # if self.reaction_conditions.set_total_flow_rate is not None:
        #     fig6a = px.scatter(x=x, y=self.reaction_conditions.set_total_flow_rates)
        #     fig6a.update_layout(title_text="Total Flow Rate")
        #     fig6a.update_xaxes(title_text=x_text) 
        #     fig6a.update_yaxes(title_text="Total Flow Rate")
        #     self.reaction_conditions.figures.append(PlotlyFigure(label='Total Flow Rate', figure=fig6a.to_plotly_json()))
        # elif self.reaction_conditions.weight_hourly_space_velocity is not None:
        #     fig6 = px.scatter(x=x, y=self.reaction_conditions.weight_hourly_space_velocity)
        #     fig6.update_layout(title_text="GHSV")
        #     fig6.update_xaxes(title_text=x_text) 
        #     fig6.update_yaxes(title_text="GHSV")
        #     self.reaction_conditions.figures.append(PlotlyFigure(label='Space Velocity', figure=fig6.to_plotly_json()))
        
        for i,c in enumerate(self.reaction_results.reactants_conversions):
                name=self.reaction_results.reactants_conversions[i].name
                fig = go.Figure()
                for j,c in enumerate(self.reaction_results.products):
                    fig.add_trace(go.Scatter(x=self.reaction_results.reactants_conversions[i].conversion, y=self.reaction_results.products[j].selectivity, name=self.reaction_results.products[j].name, mode='markers'))
                fig.update_layout(title_text="S-X plot "+ str(i), showlegend=True)
                fig.update_xaxes(title_text='Conversion '+ name ) 
                fig.update_yaxes(title_text='Selectivity')
                self.figures.append(PlotlyFigure(label='S-X plot '+ name+" Conversion", figure=fig.to_plotly_json()))



class CatalyticReaction_NH3decomposition(CatalyticReaction_core, PlotSection, EntryData):
    m_def = Section(
        label='Heterogeneous Catalysis - Activity Test NH3 Decomposition',
        categories=[UseCaseElnCategory],
    )

    data_file_h5 = Quantity(
        type=str,
        description="""
        hdf5 file that contains 'Analyzed Data' of a catalytic measurement with
        time, temperature,  Conversion, Space_time_Yield
        """,
        a_eln=dict(component='FileEditQuantity'),
        a_browser=dict(adaptor='RawFileAdaptor')
    )

    reactor_setup = SubSection(section_def=Reactor_setup)

    pretreatment = SubSection(section_def=Feed)
    reaction_conditions = SubSection(section_def=Feed)
    reaction_results = SubSection(section_def=CatalyticReactionData_core)

    measurement_details = SubSection(section_def=Measurement)

    def normalize(self, archive, logger):
        super(CatalyticReaction_NH3decomposition, self).normalize(archive, logger)

        if self.data_file_h5 is None:
            return

        if (self.data_file_h5 is not None) and (os.path.splitext(
                self.data_file_h5)[-1] != ".h5"):
            raise ValueError("Unsupported file format. This should be a hdf5 file ending with '.h5'" )
            return

        if self.data_file_h5.endswith(".h5"):
            with archive.m_context.raw_file(self.data_file_h5) as f:
                import h5py
                data = h5py.File(f.name, 'r')

        cat_data=CatalyticReactionData_core()
        feed=Feed()
        reactor_setup=Reactor_setup()
        pretreatment=Feed()
        measurement_details=Measurement()
        conversions=[]
        conversions2=[]
        rates=[]
        reagents=[]
        pre_reagents=[]
        time_on_stream=[]
        method=list(data['Analysed Data'].keys())
        for i in method:
            methodname=i
        header=data["Header"][methodname]["Header"]
        feed.catalyst_mass = header["Mass [mg]"]/1000
        feed.sampling_frequency = header["Temporal resolution [Hz]"]*ureg.hertz
        reactor_setup.reactor_volume = header["Bulk volume [mln]"]
        reactor_setup.reactor_cross_section_area = (header['Inner diameter of reactor (D) [mm]']/2)**2 * np.pi
        reactor_setup.reactor_diameter = header['Inner diameter of reactor (D) [mm]']
        reactor_setup.diluent = header['Diluent material'][0].decode()
        reactor_setup.diluent_sievefraction_high = header['Diluent Sieve fraction high [um]']
        reactor_setup.diluent_sievefraction_low = header['Diluent Sieve fraction low [um]']
        reactor_setup.catalyst_mass = header['Mass [mg]'][0]*ureg.milligram
        reactor_setup.catalyst_sievefraction_high = header['Sieve fraction high [um]']
        reactor_setup.catalyst_sievefraction_low = header['Sieve fraction low [um]']
        reactor_setup.particle_size = header['Partical size (Dp) [mm]']

        self.experimenter = header['User'][0].decode()

        pre=data["Analysed Data"][methodname]["H2 Reduction"]
        pretreatment.set_temperature = pre["Catalyst Temperature [C°]"]*ureg.celsius
        for col in pre.dtype.names :
            if col.startswith('Massflow'):
                col_split = col.split("(")
                col_split1 = col_split[1].split(")")
                if col_split1[1].startswith(' actual'): 
                    reagent = Reagent(name=col_split1[0], flow_rate=pre[col])
                    pre_reagents.append(reagent)
        pretreatment.reagents = pre_reagents
        pretreatment.flow_rates_total = pre['MassFlow (Total Gas) [mln|min]']
        number_of_runs = len(pre["Catalyst Temperature [C°]"])
        pretreatment.runs = np.linspace(0, number_of_runs - 1, number_of_runs)

        analysed=data["Analysed Data"][methodname]["NH3 Decomposition"]
        
        for col in analysed.dtype.names :
            if col.startswith('Massflow'):
                col_split = col.split("(")
                col_split1 = col_split[1].split(")")
                if col_split1[1].startswith(' actual'): 
                    reagent = Reagent(name=col_split1[0], flow_rate=analysed[col])
                    reagents.append(reagent)
        feed.reagents = reagents
        feed.flow_rates_total = analysed['MassFlow (Total Gas) [mln|min]']
        conversion = Conversion(name='NH3', conversion=np.nan_to_num(analysed['NH3 Conversion [%]']))
        conversions.append(conversion)
        conversion2 = Reactant(name='NH3', conversion=analysed['NH3 Conversion [%]'])
        conversions2.append(conversion2)
        rate = Rates(name='H2', reaction_rate=np.nan_to_num(analysed['Space Time Yield [mmolH2 gcat-1 min-1]']))
        rates.append(rate)
        feed.set_temperature = analysed['Catalyst Temperature [C°]']*ureg.celsius
        cat_data.temperature = analysed['Catalyst Temperature [C°]']*ureg.celsius
        number_of_runs = len(analysed['NH3 Conversion [%]'])
        feed.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        cat_data.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        time=analysed['Relative Time [Seconds]']
        for i in range(len(time)):
            t = float(time[i].decode("UTF-8"))-float(time[0].decode("UTF-8"))
            time_on_stream.append(t)
        cat_data.time_on_stream = time_on_stream*ureg.sec

        cat_data.reactants_conversions = conversions
        cat_data.rates = rates

        measurement_details.name = methodname
        measurement_details.datetime = pre['Date'][0].decode()

        self.reaction_results = cat_data
        self.reaction_conditions = feed
        self.reactor_setup = reactor_setup
        self.pretreatment=pretreatment
        self.measurement_details=measurement_details

        add_activity(archive)

        if conversions2 is not None:
            archive.results.properties.catalytic.reactivity.reactants = conversions2
        if cat_data.temperature is not None:
            archive.results.properties.catalytic.reactivity.test_temperatures = cat_data.temperature
        if cat_data.pressure is not None:
            archive.results.properties.catalytic.reactivity.pressure = cat_data.pressure
        # if products is not None:
        #     archive.results.properties.catalytic.reactivity.products = products
        if self.reaction_name is not None:
            archive.results.properties.catalytic.reactivity.reaction_name = self.reaction_name
            archive.results.properties.catalytic.reactivity.reaction_class = self.reaction_class

        if self.sample_reference is not None:
            if not archive.results.properties.catalytic.catalyst_characterization:
                archive.results.properties.catalytic.catalyst_characterization = CatalystCharacterization()
            if not archive.results.properties.catalytic.catalyst_synthesis:
                archive.results.properties.catalytic.catalyst_synthesis = CatalystSynthesis()

            if self.sample_reference.catalyst_type is not None:
                archive.results.properties.catalytic.catalyst_synthesis.catalyst_type = self.sample_reference.catalyst_type
            if self.sample_reference.preparation_details is not None:
                archive.results.properties.catalytic.catalyst_synthesis.preparation_method = self.sample_reference.preparation_details.preparation_method
            if self.sample_reference.surface is not None:
                archive.results.properties.catalytic.catalyst_characterization.surface_area = self.sample_reference.surface.surfacearea

        if self.sample_reference:
          if self.sample_reference.elemental_composition is not None:
            if not archive.results:
                archive.results = Results()
            if not archive.results.material:
                archive.results.material = Material()

            try:
                archive.results.material.elemental_composition = self.sample_reference.elemental_composition
        
            except Exception as e:
                logger.warn('Could not analyse elemental compostion.', exc_info=e)

        fig = px.line(x=self.reaction_results.time_on_stream, y=self.reaction_results.temperature.to('celsius'))
        fig.update_xaxes(title_text="time(h)")
        fig.update_yaxes(title_text="Temperature (°C)")
        self.figures.append(PlotlyFigure(label='figure Temp', figure=fig.to_plotly_json()))

        for i,c in enumerate(self.reaction_results.reactants_conversions):
            fig1 = px.line(x=self.reaction_results.time_on_stream, y=[self.reaction_results.reactants_conversions[i].conversion])
            fig1.update_layout(title_text="Conversion")
            fig1.update_xaxes(title_text="time(h)")
            fig1.update_yaxes(title_text="Conversion (%)")
            self.figures.append(PlotlyFigure(label='figure Conversion', figure=fig1.to_plotly_json()))

        fig2 = px.line(x=self.reaction_results.temperature.to('celsius'), y=[self.reaction_results.rates[0].reaction_rate])
        fig2.update_xaxes(title_text="Temperature (°C)")
        fig2.update_yaxes(title_text="reaction rate (mmol(H2)/gcat/min)")
        self.figures.append(PlotlyFigure(label='figure rates', figure=fig2.to_plotly_json()))

        fig3 = px.scatter(x=self.pretreatment.runs, y=self.pretreatment.set_temperature.to('celsius'))
        fig3.update_layout(title_text="Temperature")
        fig3.update_xaxes(title_text="measurement points",) 
        fig3.update_yaxes(title_text="Temperature (°C)")
        self.pretreatment.figures.append(PlotlyFigure(label='Temperature', figure=fig3.to_plotly_json()))

        fig4 = px.scatter(x=self.reaction_conditions.runs, y=self.reaction_conditions.set_temperature.to('celsius'))
        fig4.update_layout(title_text="Temperature")
        fig4.update_xaxes(title_text="measurement points",) 
        fig4.update_yaxes(title_text="Temperature (°C)")
        self.reaction_conditions.figures.append(PlotlyFigure(label='Temperature', figure=fig4.to_plotly_json()))

m_package.__init_metainfo__()
