import numpy as np
import os

from nomad.metainfo import (
    Quantity,
    Section,
    SubSection,
    Package)

from nomad.units import ureg
from ase.data import chemical_symbols

# from nomad.datamodel.metainfo.eln import Measurement

from nomad.datamodel.metainfo.basesections import CompositeSystem, Measurement, CompositeSystemReference

from nomad.datamodel.data import ArchiveSection

from nomad.datamodel.results import (Results, Material, Properties, CatalyticProperties,
                                     CatalystCharacterization, CatalystSynthesis)
from nomad.datamodel.results import Product as Product_result
from nomad.datamodel.results import Reactant as Reactant_result

from nomad.datamodel.data import EntryData, UseCaseElnCategory

from .catalytic_measurement import (
    CatalyticReactionData, CatalyticReactionData_core, Conversion, Rates, ReactorSetup, ReactionConditions, ReactionConditionsSimple,
    add_activity
    )

from .catalytic_measurement import Product as Product_data
from .catalytic_measurement import Reagent as Reagent_data
from .catalytic_measurement import Reactant as Reactant_data

from nomad.datamodel.metainfo.plot import PlotSection, PlotlyFigure
import plotly.express as px
import plotly.graph_objs as go

from nomad.datamodel.metainfo.annotations import ELNAnnotation

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
        a_eln=dict(component='EnumEditQuantity',
        # props=dict(
        #     suggestions=['A. Trunschke',
        #                  'C. Hess',
        #                  'M. Behrens',
        #                  'unknown'])
        ),
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
    m_def = Section(label_quantity='method_surface_area_determination', a_eln=ELNAnnotation(label='Surface Area'))

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
          A description of the method used to measure the surface area of the sample.
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
    m_def = Section(
        label='Heterogeneous Catalysis - Catalyst Sample',
        #a_eln=dict(hide=['cas_uri', 'cas_number', 'cas_name', 'inchi', 'inchi_key',
        #                 'smile', 'canonical_smile', 'cas_synonyms', 'molecular mass']),
        categories=[UseCaseElnCategory],
    )

    preparation_details = SubSection(section_def=Preparation)

    surface = SubSection(section_def=SurfaceArea, a_eln=ELNAnnotation(label='Surface Area'))

    storing_institution = Quantity(
        type=str,
        shape=[],
        description="""
        The institution at which the sample is stored.
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(
            suggestions=['Fritz-Haber-Institut Berlin / Abteilung AC',
                         'Fritz-Haber-Institut Berlin / ISC', 'TU Berlin / BasCat']))
    )

    catalyst_type = Quantity(
        type=str,
        shape=[],
        description="""
          A classification of the catalyst type.
          """,
        a_eln=dict(
            component='EnumEditQuantity', props=dict(
                suggestions=['bulk catalyst', 'supported catalyst', 'single crystal','metal','oxide',
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
        if self.preparation_details is not None:
            archive.results.properties.catalytic.catalyst_synthesis.preparation_method = self.preparation_details.preparation_method

    ### testing how to add referenced methods to results#####:
        # methods=['XRF', 'XRD', 'XPS']
        # if self.    
            # archive.results.properties.catalytic.catalyst_characterization.method=methods

class ReactorFilling(ArchiveSection):
    m_def = Section(description='A class containing information about the catalyst and filling in the reactor.', 
                    label='Catalyst')

    catalyst_name = Quantity(
        type=str, shape=[], a_eln=ELNAnnotation(component='StringEditQuantity'))

    sample_reference = Quantity(
        type=CatalystSample, description='A reference to the sample used for the measurement.',
        a_eln=ELNAnnotation(component='ReferenceEditQuantity', label='Entity Reference'))

    catalyst_mass = Quantity(
        type=np.float64, shape=[], unit='mg', a_eln=ELNAnnotation(component='NumberEditQuantity'))

    catalyst_density = Quantity(
        type=np.float64, shape=[], unit='g/mL', a_eln=ELNAnnotation(component='NumberEditQuantity'))

    apparent_catalyst_volume = Quantity(
        type=np.float64, shape=[], unit='mL', a_eln=ELNAnnotation(component='NumberEditQuantity'))

    catalyst_sievefraction_upper_limit = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))
    catalyst_sievefraction_lower_limit = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))
    particle_size = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))
    diluent = Quantity(
        type=str,
        shape=[],
        description="""
        A component that is mixed with the catalyst to dilute and prevent transport
        limitations and hot spot formation.
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(
            suggestions=['SiC', 'SiO2', 'unknown']))
    )
    diluent_sievefraction_upper_limit = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))
    diluent_sievefraction_lower_limit = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))

    def normalize(self, archive, logger):
        super(ReactorFilling, self).normalize(archive, logger)

        if self.sample_reference is None:
            if self.m_root().data.samples != []:
                self.sample_reference = self.m_root().data.samples[0].reference
        if self.sample_reference is not None:
            if self.m_root().data.samples == []:
                sample1_reference = CompositeSystemReference(reference=self.sample_reference)
                self.m_root().data.samples.append(sample1_reference)
            elif self.m_root().data.samples[0].reference is None:
                self.m_root().data.samples[0].reference = self.sample_reference
            self.sample_reference.normalize(archive, logger)

        if self.catalyst_name is None and self.sample_reference is not None:
            self.catalyst_name = self.sample_reference.name

        if self.apparent_catalyst_volume is None and self.catalyst_mass is not None and self.catalyst_density is not None:
            self.apparent_catalyst_volume = self.catalyst_mass / self.catalyst_density


# class Feed(ReactionConditions, ArchiveSection):  
#     m_def = Section(description='A class containing information about the feed gas and reactor filling including the catalyst.')

#     reactor_filling = SubSection(section_def=ReactorFilling)
    
#     def normalize(self, archive, logger):
#         super(Feed, self).normalize(archive, logger)


class CatalyticReaction_core(Measurement, ArchiveSection):

    sample_reference = Quantity(
        type=CatalystSample,
        description="""
        The link to the entry of the catalyst sample used in the experiment.
        """,
        a_eln=dict(component='ReferenceEditQuantity')
    )

    reaction_class = Quantity(
        type=str,
        description="""
        A highlevel classification of the studied reaction.
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(suggestions=[
            'Oxidation', 'Hydrogenation', 'Dehydrogenation', 'Cracking', 'Isomerisation', 'Coupling']
        )),
        iris=['https://w3id.org/nfdi4cat/voc4cat_0007010'])

    reaction_name = Quantity(
        type=str,
        description="""
        The name of the studied reaction.
        """,
        a_eln=dict(
            component='EnumEditQuantity', props=dict(suggestions=[
                'Alkane Oxidation', 'Oxidation of Ethane', 'Oxidation of Propane',
                'Oxidation of Butane', 'CO hydrogenation', 'Methanol Synthesis', 'Fischer-Tropsch',
                'Water gas shift reaction', 'Ammonia Synthesis', 'Ammonia decomposition'])),
        iris=['https://w3id.org/nfdi4cat/voc4cat_0007009'])


    experiment_handbook = Quantity(
        description="""
        In case the experiment was performed according to a handbook.
        """,
        type=str,
        shape=[],
        a_eln=dict(component='FileEditQuantity')
    )

    institute = Quantity(
        type=str,
        shape=[],
        description="""
        The institution at which the measurement was performed.
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
        The person that performed or started the measurement.
        """,
        a_eln=dict(component='EnumEditQuantity')
    )


class SimpleCatalyticReaction(CatalyticReaction_core, EntryData):
    m_def = Section(
        label='Heterogeneous Catalysis - Simple Catalytic Reaction for measurment plugin',
        categories=[UseCaseElnCategory]
    )
    reaction_condition = SubSection(section_def=ReactionConditionsSimple, a_eln=ELNAnnotation(label='Reaction Conditions'))
    reactor_filling = SubSection(section_def=ReactorFilling)
    reactor_setup = SubSection(section_def=ReactorSetup)
    reaction_results = SubSection(section_def=CatalyticReactionData_core, a_eln=ELNAnnotation(label='Reaction Results'))


class CatalyticReaction(CatalyticReaction_core, PlotSection, EntryData):
    """
    This schema is originally adapted to map the data of the clean Oxidation dataset (JACS,
    https://doi.org/10.1021/jacs.2c11117) The descriptions in the quantities
    represent the instructions given to the user who manually curated the data. The schema has since been extendet to match other, similar datasets, with multiple products.
    """
    m_def = Section(
        label='Heterogeneous Catalysis - Activity Test Clean Data',
        a_eln=ELNAnnotation(properties=dict(order= ['name','data_file', 'sample_reference','reaction_name','reaction_class',
                            'experimenter', 'institute', 'experiment_handbook'])),
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

    reactor_setup = SubSection(section_def=ReactorSetup)
    reactor_filling = SubSection(section_def=ReactorFilling)

    reaction_conditions = SubSection(section_def=ReactionConditions, a_eln=ELNAnnotation(label='Reaction Conditions'))
    reaction_results = SubSection(section_def=CatalyticReactionData, a_eln=ELNAnnotation(label='Reaction Results'))

    def normalize(self, archive, logger):
        super(CatalyticReaction, self).normalize(archive, logger)
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
        feed = ReactionConditions()
        reactor_filling = ReactorFilling()
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
            
            if col_split[0] == 'step':
                feed.runs = data['step']
                cat_data.runs = data['step']

            if col_split[0] == "x":
                reagent = Reagent_data(name=col_split[1], gas_concentration_in=data[col])
                reagent_names.append(col_split[1])
                reagents.append(reagent)
            if col_split[0] == "mass":
                catalyst_mass_vector = data[col]
                if '(g)' in col_split[1]: 
                    reactor_filling.catalyst_mass = catalyst_mass_vector[0]*ureg.gram
                else:
                    reactor_filling.catalyst_mass = catalyst_mass_vector[0]*ureg.milligram
            if col_split[0] == "temperature":
                if "K" in col_split[1]:
                    cat_data.temperature = np.nan_to_num(data[col])
                else:
                    cat_data.temperature = np.nan_to_num(data[col])*ureg.celsius

            if col_split[0] == "TOS":
                cat_data.time_on_stream = data[col]
                feed.time_on_stream = data[col]

            if col_split[0] == "C-balance":
                cat_data.c_balance = np.nan_to_num(data[col])

            if col_split[0] == "GHSV":
                feed.gas_hourly_space_velocity = np.nan_to_num(data[col])

            if col_split[0] == "Vflow":
                feed.set_total_flow_rate = np.nan_to_num(data[col])

            if col_split[0] == "set_pressure":	
                feed.set_pressure = np.nan_to_num(data[col])
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
                    conversion2 = Reactant_data(name=col_split[1], conversion=np.nan_to_num(data[col]), gas_concentration_in=(np.nan_to_num(data['x '+col_split[1]+' (%)'])))
                    conversions2.append(conversion2)
                except KeyError:
                    pass
                try:
                    conversion2 = Reactant_data(name=col_split[1], conversion=np.nan_to_num(data[col]), gas_concentration_in=np.nan_to_num(data['x '+col_split[1]])*100)
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

            if col_split[0] == "y":  # concentration out
                if col_split[1] in reagent_names:
                    conversion2 = Reactant_data(name=col_split[1], gas_concentration_in=np.nan_to_num(data['x '+col_split[1]+' (%)']), gas_concentration_out=np.nan_to_num(data[col]), conversion=np.nan_to_num((1-(data[col]/data['x '+col_split[1]+' (%)']))*100))
                    conversions2.append(conversion2)
                else:
                    product = Product_data(name=col_split[1], gas_concentration_out=np.nan_to_num(data[col]))
                    products.append(product)
                    product_names.append(col_split[1])	
            
            if col_split[0] == "S_p":  # selectivity
                product = Product_data(name=col_split[1], selectivity=np.nan_to_num(data[col]))
                for i, p in enumerate(products):
                    if p.name == col_split[1]:
                        product = products.pop(i)
                        product.selectivity = np.nan_to_num(data[col])
                        break
                products.append(product)
                product_names.append(col_split[1])

        for reagent in reagents:
            reagent.normalize(archive, logger)
        feed.reagents = reagents
        
        if cat_data.runs is None:
            cat_data.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        cat_data.products = products
        if conversions != []:
            cat_data.reactants_conversions = conversions
        elif conversions2 != []:
            print('in if clause', conversions2)
            cat_data.reactants_conversions = conversions2
        cat_data.rates = rates
        self.reaction_conditions = feed
        self.reaction_results = cat_data
        if self.reactor_filling is None and reactor_filling is not None:
            self.reactor_filling = reactor_filling
        
        self.reaction_results.normalize(archive, logger)

        conversions_results = []
        for i in conversions2:
            if i.name in ['He', 'helium', 'Ar', 'argon', 'inert']:
                continue
            else:
                for j in reagents:
                    if i.name == j.name:
                        if j.pure_component.iupac_name is not None:
                            i.name = j.pure_component.iupac_name
                            react = Reactant_result(name=i.name, conversion=i.conversion, gas_concentration_in=i.gas_concentration_in, gas_concentration_out=i.gas_concentration_out)
                            conversions_results.append(react)
        product_results=[]
        for i in products:
            if i.pure_component is not None:
                if i.pure_component.iupac_name is not None:
                    i.name = i.pure_component.iupac_name
            prod = Product_result(name=i.name, selectivity=i.selectivity, gas_concentration_out=i.gas_concentration_out)
            product_results.append(prod)

        add_activity(archive)

        if conversions_results is not None:
            archive.results.properties.catalytic.reaction.reactants = conversions_results
        if cat_data.temperature is not None:
            archive.results.properties.catalytic.reaction.temperature = cat_data.temperature
        if cat_data.temperature is None and feed.set_temperature is not None:
            archive.results.properties.catalytic.reaction.temperature = feed.set_temperature
        if cat_data.pressure is not None:
            archive.results.properties.catalytic.reaction.pressure = cat_data.pressure
        elif feed.set_pressure is not None:
            archive.results.properties.catalytic.reaction.pressure = feed.set_pressure
        if feed.gas_hourly_space_velocity is not None:
            archive.results.properties.catalytic.reaction.gas_hourly_space_velocity = feed.gas_hourly_space_velocity
        if products is not None:
            archive.results.properties.catalytic.reaction.products = product_results
        if self.reaction_name is not None:
            archive.results.properties.catalytic.reaction.name = self.reaction_name
            archive.results.properties.catalytic.reaction.type = self.reaction_class

        if self.sample_reference is not None:
            add_catalyst(archive)
            
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
                for i in self.sample_reference.elemental_composition:    
                    if i.element not in chemical_symbols:
                        logger.warn(
                            f"'{self.sample_reference.elemental_composition.element}' is not a valid element symbol and this "
                            'elemental_composition section will be ignored.'
                        )
                    elif i.element not in archive.results.material.elements:
                        archive.results.material.elements += [i.element]
        
        ###Figures definitions###
        self.figures = []
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
        
        if cat_data.pressure is not None or feed.set_pressure is not None:
            figP = go.Figure()
            if cat_data.pressure is not None:
                figP = px.line(x=x, y=cat_data.pressure.to("bar"))
            elif feed.set_pressure is not None:
                figP = px.line(x=x, y=feed.set_pressure.to("bar"))
            figP.update_xaxes(title_text=x_text)
            figP.update_yaxes(title_text="Pressure (bar)")
            self.figures.append(PlotlyFigure(label='figure Pressure', figure=figP.to_plotly_json()))
        
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
            # try:
            #     fig2 = px.line(x=self.reaction_results.temperature.to('celsius'), y=[self.reaction_results.rates[0].reaction_rate])
            #     fig2.update_xaxes(title_text="Temperature (°C)")
            #     fig2.update_yaxes(title_text="reaction rate (mmol(H2)/gcat/min)")
            #     self.figures.append(PlotlyFigure(label='figure rates', figure=fig2.to_plotly_json()))
            # except:
            #     print("No rates defined")

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
        hide=['description',],
        a_eln=ELNAnnotation(properties=dict(order= ['name','data_file_h5', 'sample_reference','reaction_name','reaction_class',
                            'experimenter', 'institute', 'experiment_handbook'])),
        categories=[UseCaseElnCategory],
    )

    data_file_h5 = Quantity(
        type=str,
        description="""
        hdf5 file that contains 'Sorted Data' of a catalytic measurement with
        time, temperature,  Conversion, Space_time_Yield
        """,
        a_eln=dict(component='FileEditQuantity'),
        a_browser=dict(adaptor='RawFileAdaptor')
    )

    reactor_setup = SubSection(section_def=ReactorSetup)
    reactor_filling = SubSection(section_def=ReactorFilling)

    pretreatment = SubSection(section_def=ReactionConditions)
    reaction_conditions = SubSection(section_def=ReactionConditions)
    reaction_results = SubSection(section_def=CatalyticReactionData_core)

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
        feed=ReactionConditions()
        reactor_setup=ReactorSetup()
        reactor_filling=ReactorFilling()
        pretreatment=ReactionConditions()
        sample=CompositeSystemReference()
        conversions=[]
        conversions2=[]
        rates=[]
        reagents=[]
        pre_reagents=[]
        time_on_stream=[]
        time_on_stream_reaction=[]
        method=list(data['Sorted Data'].keys())
        for i in method:
            methodname=i
        header=data["Header"][methodname]["Header"]
        reactor_filling.catalyst_mass = header["Catalyst Mass [mg]"]/1000
        feed.sampling_frequency = header["Temporal resolution [Hz]"]*ureg.hertz
        reactor_setup.name = 'Haber'
        reactor_setup.reactor_type = 'plug flow reactor'
        reactor_setup.reactor_volume = header["Bulk volume [mln]"]
        reactor_setup.reactor_cross_section_area = (header['Inner diameter of reactor (D) [mm]']/2)**2 * np.pi
        reactor_setup.reactor_diameter = header['Inner diameter of reactor (D) [mm]']
        reactor_filling.diluent = header['Diluent material'][0].decode()
        reactor_filling.diluent_sievefraction_upper_limit = header['Diluent Sieve fraction high [um]']
        reactor_filling.diluent_sievefraction_lower_limit = header['Diluent Sieve fraction low [um]']
        reactor_filling.catalyst_mass = header['Catalyst Mass [mg]'][0]*ureg.milligram
        reactor_filling.catalyst_sievefraction_upper_limit = header['Sieve fraction high [um]']
        reactor_filling.catalyst_sievefraction_lower_limit = header['Sieve fraction low [um]']
        reactor_filling.particle_size = header['Particle size (Dp) [mm]']

        self.experimenter = header['User'][0].decode()

        pre=data["Sorted Data"][methodname]["H2 Reduction"]
        pretreatment.set_temperature = pre["Catalyst Temperature [C°]"]*ureg.celsius
        for col in pre.dtype.names :
            if col == 'Massflow3 (H2) Target Calculated Realtime Value [mln|min]':
                pre_reagent = Reagent_data(name='hydrogen', flow_rate=pre[col])
                pre_reagents.append(pre_reagent)
            if col == 'Massflow5 (Ar) Target Calculated Realtime Value [mln|min]':
                pre_reagent = Reagent_data(name='argon', flow_rate=pre[col])
                pre_reagents.append(pre_reagent)
            # if col.startswith('Massflow'):
            #     col_split = col.split("(")
            #     col_split1 = col_split[1].split(")")
            #     if col_split1[1].startswith(' actual'): 
            #         reagent = Reagent(name=col_split1[0], flow_rate=pre[col])
            #         pre_reagents.append(reagent)
        pretreatment.reagents = pre_reagents
        pretreatment.set_total_flow_rate = pre['Target Total Gas (After Reactor) [mln|min]']
        number_of_runs = len(pre["Catalyst Temperature [C°]"])
        pretreatment.runs = np.linspace(0, number_of_runs - 1, number_of_runs)

        time=pre['Relative Time [Seconds]']
        for i in range(len(time)):
            t = float(time[i].decode("UTF-8"))-float(time[0].decode("UTF-8"))
            time_on_stream.append(t)
        pretreatment.time_on_stream = time_on_stream*ureg.sec

        analysed=data["Sorted Data"][methodname]["NH3 Decomposition"]
        
        for col in analysed.dtype.names :
            if col.endswith('Target Calculated Realtime Value [mln|min]'):
                name_split=col.split("(")
                gas_name=name_split[1].split(")")
                if 'NH3' in gas_name:
                    reagent = Reagent_data(name='NH3', flow_rate=analysed[col])
                    reagents.append(reagent)
                else:
                    reagent = Reagent_data(name=gas_name[0], flow_rate=analysed[col])
                    reagents.append(reagent)
        feed.reagents = reagents
        # feed.flow_rates_total = analysed['MassFlow (Total Gas) [mln|min]']
        conversion = Conversion(name='ammonia', conversion=np.nan_to_num(analysed['NH3 Conversion [%]']))
        conversions.append(conversion)
        conversion2 = Reactant_result(name='ammonia', conversion=analysed['NH3 Conversion [%]'])
        # conversion2.conversion = analysed['NH3 Conversion [%]'].reshape(-1,10).mean(axis=1)  ## trying to reduce size of array
        conversions2.append(conversion2)
        rate = Rates(name='molecular hydrogen', reaction_rate=np.nan_to_num(analysed['Space Time Yield [mmolH2 gcat-1 min-1]']*ureg.mmol/ureg.g/ureg.minute))
        rates.append(rate)
        feed.set_temperature = analysed['Catalyst Temperature [C°]']*ureg.celsius
        cat_data.temperature = analysed['Catalyst Temperature [C°]']*ureg.celsius
        number_of_runs = len(analysed['NH3 Conversion [%]'])
        feed.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        cat_data.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        time=analysed['Relative Time [Seconds]']
        for i in range(len(time)):
            t = float(time[i].decode("UTF-8"))-float(time[0].decode("UTF-8"))
            time_on_stream_reaction.append(t)
        cat_data.time_on_stream = time_on_stream_reaction*ureg.sec

        cat_data.reactants_conversions = conversions
        cat_data.rates = rates

        self.method = methodname
        self.datetime = pre['Date'][0].decode()

        sample.reference = self.sample_reference
        sample.name = 'catalyst'
        sample.lab_id = str(data["Header"]["Header"]['SampleID'][0])

        self.reaction_results = cat_data
        self.reaction_conditions = feed
        self.reactor_setup = reactor_setup
        self.pretreatment=pretreatment
        self.reactor_filling=reactor_filling

        self.samples.append(sample)

        add_activity(archive)

        
        products_results = []
        for i in ['molecular nitrogen', 'molecular hydrogen']:
            product = Product_result(name=i)
            products_results.append(product)
        self.products = products_results

        if conversions2 is not None:
            archive.results.properties.catalytic.reaction.reactants = conversions2
        if cat_data.temperature is not None:
            archive.results.properties.catalytic.reaction.temperatures = cat_data.temperature
        if cat_data.pressure is not None:
            archive.results.properties.catalytic.reaction.pressure = cat_data.pressure
        if products_results != []:
            archive.results.properties.catalytic.reaction.products = products_results
        if rates is not None:
            archive.results.properties.catalytic.reaction.rates = rates
        if self.reaction_name is not None:
            archive.results.properties.catalytic.reaction.name = self.reaction_name
            archive.results.properties.catalytic.reaction.type = self.reaction_class

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
                archive.results.material.elements = self.sample_reference.elemental_composition.elements
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
