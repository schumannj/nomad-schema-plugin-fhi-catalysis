import numpy as np
import os

from ase.data import chemical_symbols

from nomad.metainfo import (
    Quantity,
    Section,
    SubSection,
    Package)

from nomad.datamodel.metainfo.eln import (
    #CompositeSystem,
    Measurement)

from nomad.datamodel.metainfo.basesections import CompositeSystem, System

from nomad.datamodel.data import ArchiveSection

from nomad.datamodel.results import (Results, Material, Properties, CatalyticProperties,
                                     CatalystCharacterization, CatalystSynthesis, Reactivity)
from nomad.datamodel.data import EntryData, UseCaseElnCategory

from .catalytic_measurement import (
    CatalyticReactionData, CatalyticReactionData_core, Feed, Reagent, Conversion, Rates, Reactor_setup,
    Pretreatment,
    ECatalyticReactionData, PotentiostaticMeasurement )

from nomad.datamodel.results import Product, Reactant

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


def add_activity(archive):
    '''Adds metainfo structure for catalysis activity test data.'''
    if not archive.results:
        archive.results = Results()
    if not archive.results.properties:
        archive.results.properties = Properties()
    if not archive.results.properties.catalytic:
        archive.results.properties.catalytic = CatalyticProperties()
    if not archive.results.properties.catalytic.reactivity:
        archive.results.properties.catalytic.reactivity = Reactivity()


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


class Surface_Area(ArchiveSection):
    m_def = Section(label_quantity='method_surface_area_determination')

    surfacearea = Quantity(
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
        super(Surface_Area, self).normalize(archive, logger)

        add_catalyst(archive)

        # if self.method_surface_area is not None:
        archive.results.properties.catalytic.catalyst_characterization.surface_area = self.surfacearea
        archive.results.properties.catalytic.catalyst_characterization.method_surface_area = self.method_surface_area_determination


class CatalystSample(System, EntryData):
    """
    This schema is adapted to map the data of the clean Oxidation dataset (JACS,
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

    surface_area = SubSection(section_def=Surface_Area)

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
        if self.surface_area is not None:
            archive.results.properties.catalytic.catalyst_characterization.surface_area = self.surface_area.surfacearea
            archive.results.properties.catalytic.catalyst_characterization.method_surface_area = self.surface_area.method_surface_area_determination
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
            'Oxidation', 'Hydrogenation', 'Isomerisation', 'Coupling']
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


class CatalyticReaction(CatalyticReaction_core, EntryData):

    m_def = Section(
        label='Heterogeneous Catalysis - Activity Test',
        categories=[UseCaseElnCategory],
        a_plot=[{
            "label": "Selectivity [%]",
            'x': 'reaction_data/runs',
            'y': ['reaction_data/products/:/selectivity'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "Reaction Rates [mmol/g/hour]",
            'x': 'reaction_data/runs',
            'y': ['reaction_data/rates/:/reaction_rate'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "Reaction Conditions",
            'x': 'reaction_data/runs',
            'y': ['feed/reagents/:/gas_concentration_in'],'y2': ['reaction_data/temperature'], 
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}}]
    )

    data_file = Quantity(
        type=str,
        description="""
        excel or csv file that contains results of a catalytic measurement with
        temperature, (pressure,) gas feed composition, yield, rates and selectivities
        """,
        a_eln=dict(component='FileEditQuantity'),
        a_browser=dict(adaptor='RawFileAdaptor'))

    reactor_setup = SubSection(section_def=Reactor_setup)

    feed = SubSection(section_def=Feed)
    reaction_data = SubSection(section_def=CatalyticReactionData)

    measurement_details = SubSection(section_def=Measurement)

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
                cat_data.temperature = np.nan_to_num(data[col])

            if col_split[0] == "TOS":
                cat_data.time_on_stream = data[col]

            if col_split[0] == "C-balance":
                cat_data.c_balance = np.nan_to_num(data[col])

            if col_split[0] == "GHSV":
                feed.space_velocity = np.nan_to_num(data[col])

            if col_split[0] == "Vflow":
                feed.flow_rates_total = np.nan_to_num(data[col])

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
                    conversion2 = Reactant(name=col_split[1], conversion=np.nan_to_num(data[col]), gas_concentration_in=(data['x '+col_split[1]+' (%)'])/100)
                    conversions2.append(conversion2)
                except KeyError:
                    pass
                try:
                    conversion2 = Reactant(name=col_split[1], conversion=np.nan_to_num(data[col]), gas_concentration_in=data['x '+col_split[1]])
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
        feed.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        cat_data.products = products
        cat_data.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        cat_data.reactants_conversions = conversions
        cat_data.rates = rates
        self.feed = feed
        self.reaction_data = cat_data

        add_activity(archive)

        if conversions2 is not None:
            archive.results.properties.catalytic.reactivity.reactants = conversions2
        if cat_data.temperature is not None:
            archive.results.properties.catalytic.reactivity.test_temperatures = cat_data.temperature
        if cat_data.pressure is not None:
            archive.results.properties.catalytic.reactivity.pressure = cat_data.pressure
        if feed.space_velocity is not None:
            archive.results.properties.catalytic.reactivity.gas_hourly_space_velocity = feed.space_velocity
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
            if self.sample_reference.surface_area is not None:
                archive.results.properties.catalytic.catalyst_characterization.surface_area = self.sample_reference.surface_area.surfacearea

        if self.sample_reference.elemental_composition is not None:
            if not archive.results:
                archive.results = Results()
            if not archive.results.material:
                archive.results.material = Material()

            try:
                archive.results.material.elemental_composition = self.sample_reference.elemental_composition
                # if self.sample_reference.elemental_composition.element not in chemical_symbols:
                #     logger.warn(
                #         f"'{self.sample_reference.elemental_composition.element}' is not a valid element symbol and this "
                #         "elemental_composition section will be ignored.")
                # elif self.sample_reference.elemental_composition.element not in archive.results.material.elements:
                #     archive.results.material.elements += [self.sample_reference.elemental_composition.element]

            except Exception as e:
                logger.warn('Could not analyse elemental compostion.', exc_info=e)


class CatalyticReaction_NH3decomposition(CatalyticReaction_core, EntryData):
    m_def = Section(
        label='Heterogeneous Catalysis - Activity Test NH3 Decomposition',
        categories=[UseCaseElnCategory],
        a_plot=[{
            "label": "Conversion [%]",
            'x': 'reaction_data/time_on_stream',
            'y': ['reaction_data/reactants_conversions/:/conversion'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "Space Time Yield [mmol/g/min]",
            'x': 'reaction_data/temperature',
            'y': ['reaction_data/rates/:/reaction_rate'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}}]
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

    pretreatment = SubSection(section_def=Pretreatment)
    feed = SubSection(section_def=Feed)
    reaction_data = SubSection(section_def=CatalyticReactionData_core)

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
        pretreatment=Pretreatment()
        measurement_details=Measurement()
        conversions=[]
        conversions2=[]
        rates=[]
        reagents=[]
        pre_reagents=[]
        time_in_hours=[]
        method=list(data['Analysed Data'].keys())
        for i in method:
            methodname=i
        header=data["Header"][methodname]["Header"]
        feed.catalyst_mass = header["Mass [mg]"]/1000
        feed.sampling_frequency = header["Temporal resolution [Hz]"]
        reactor_setup.reactor_volume = header["Bulk volume [mln]"]
        reactor_setup.reactor_cross_section_area = (header['Inner diameter of reactor (D) [mm]']/2)**2 * np.pi
        reactor_setup.reactor_diameter = header['Inner diameter of reactor (D) [mm]']
        reactor_setup.diluent = header['Diluent material'][0].decode()
        reactor_setup.diluent_sievefraction_high = header['Diluent Sieve fraction high [um]']
        reactor_setup.diluent_sievefraction_low = header['Diluent Sieve fraction low [um]']
        reactor_setup.catalyst_mass = header['Mass [mg]'][0]/1000
        reactor_setup.catalyst_sievefraction_high = header['Sieve fraction high [um]']
        reactor_setup.catalyst_sievefraction_low = header['Sieve fraction low [um]']
        reactor_setup.particle_size = header['Partical size (Dp) [mm]']

        self.experimenter = header['User'][0].decode()

        pre=data["Analysed Data"][methodname]["H2 Reduction"]
        pretreatment.set_temperature = pre["Catalyst Temperature [C°]"]
        for col in pre.dtype.names :
            if col.startswith('Massflow'):
                col_split = col.split("(")
                col_split1 = col_split[1].split(")")
                if col_split1[1].startswith(' actual'): 
                    reagent = Reagent(name=col_split1[0], flow_rate=pre[col])
                    pre_reagents.append(reagent)
        pretreatment.reagents = pre_reagents
        pretreatment.flow_rates_total = pre['MassFlow (Total Gas) [mln|min]']
        
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
        cat_data.temperature = analysed['Catalyst Temperature [C°]']
        number_of_runs = len(analysed['NH3 Conversion [%]'])
        cat_data.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        time=analysed['Relative Time [Seconds]']
        for i in range(len(time)):
            t = (float(time[i].decode('UTF-8'))-float(time[0].decode('UTF-8')))/3600
            time_in_hours.append(t)

        cat_data.time_on_stream = time_in_hours

        cat_data.reactants_conversions = conversions
        cat_data.rates = rates

        measurement_details.name = methodname
        measurement_details.datetime = pre['Date'][0].decode()


        self.reaction_data = cat_data
        self.feed = feed
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
            if self.sample_reference.surface_area is not None:
                archive.results.properties.catalytic.catalyst_characterization.surface_area = self.sample_reference.surface_area.surfacearea

        if self.sample_reference.elemental_composition is not None:
            if not archive.results:
                archive.results = Results()
            if not archive.results.material:
                archive.results.material = Material()

            try:
                archive.results.material.elemental_composition = self.sample_reference.elemental_composition
                # if self.sample_reference.elemental_composition.element not in chemical_symbols:
                #    logger.warn(
                #        f"'{self.sample_reference.elemental_composition.element}' is not a valid element symbol and this "
                #        "elemental_composition section will be ignored.")
                # elif self.sample_reference.elemental_composition.element not in archive.results.material.elements:
                #    archive.results.material.elements += [self.sample_reference.elemental_composition.element]

            except Exception as e:
                logger.warn('Could not analyse elemental compostion.', exc_info=e)


class ElectrocatalystSample(CatalystSample, EntryData):

    m_def = Section(
        label='Electrocatalyst Sample',
        # a_eln=dict(  # lane_width='400px',
        #     hide=['cas_uri', 'cas_number', 'cas_name', 'inchi', 'inchi_key', 'smile',
        #           'canonical_smile', 'cas_synonyms', 'molecular mass']),
        categories=[UseCaseElnCategory])

    electrochemically_active_surface_area = Quantity(
        type=np.float64,
        unit=("m**2/g"),
        a_eln=dict(
            component='NumberEditQuantity', defaultDisplayUnit='m**2/g',
        ))

    electrode_area_geometric = Quantity(
        type=np.dtype(np.float64),
        unit=("cm**2"),
        a_eln=dict(
            component='NumberEditQuantity', defaultDisplayUnit='cm**2',
        ))

    catalyst_mass = Quantity(
        type=np.dtype(np.float64),
        unit=("mg"),
        description="""
        loading of catalyst on electrode support
        """,
        a_eln=dict(
            component='NumberEditQuantity', defaultDisplayUnit='mg',
        ))

    sample_support = Quantity(
        type=str,
        shape=[],
        description="""
          electrode support
          """,
        a_eln=dict(
            component='EnumEditQuantity', props=dict(
                suggestions=['carbon paper', 'unknown'])))

    sample_binder = Quantity(
        type=str,
        shape=[],
        description="""
          binder to fix catalyst to support
          """,
        a_eln=dict(
            component='EnumEditQuantity', props=dict(
                suggestions=['Nafion', 'unknown'])))

    def normalize(self, archive, logger):
        super(ElectrocatalystSample, self).normalize(archive, logger)


class ElectroCatalyticReaction(EntryData):  # used to be MSection

    m_def = Section(
        label='Electrocatalytic Measurement',
        categories=[UseCaseElnCategory],
    )

    sample_reference = Quantity(
        type=ElectrocatalystSample,
        description="""
        link to an electrocatalyst sample entry
        """,
        a_eln=dict(component='ReferenceEditQuantity')
    )

    reaction_class = Quantity(
        type=str,
        description="""
        highlevel classification of reaction
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(suggestions=[
            'Oxidation', 'Reduction']
        )))

    reaction_name = Quantity(
        type=str,
        description="""
          name of reaction
          """,
        a_eln=dict(
            component='EnumEditQuantity', props=dict(suggestions=[
                'CO2RR', 'HER', 'OER'])))

    existing_experiment_handbook = Quantity(
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
        person that performed or started the measurement
        """,
        a_eln=dict(component='EnumEditQuantity')
    )

    data_file = Quantity(
        type=str,
        description="""
        An excel or csv file that contains results of a electrocatalytic measurement with
        corrected potential, current, gas feed composition, partial current density,
        and Faradeic Efficiency (product selectivities)
        """,
        a_eln=dict(component='FileEditQuantity'),
        a_browser=dict(adaptor='RawFileAdaptor'))

    electrolyte = Quantity(
        type=str,
        a_eln=dict(component='StringEditQuantity',
                   props=dict(suggestions=['Water', 'Base', 'Acid'])))

    electrolyte_concentration = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='mol/L',
        a_eln=dict(
            component='NumberEditQuantity', defaultDisplayUnit='mol/L'))

    electrolyte_gas_saturation = Quantity(
        type=str,
        a_eln=dict(component='StringEditQuantity',
                   props=dict(suggestions=['Argon', 'N2', 'CO2'])))

    stirring_speed = Quantity(
        type=np.float64,
        shape=[],
        unit='1/s',
        a_eln=dict(
            component='NumberEditQuantity', defaultDisplayUnit='1/s'))

    pH = Quantity(
        type=np.dtype(np.float64),
        unit=("pH"),
        a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='pH'))

    measurement_type = Quantity(
        type=str,
        shape=[],
        a_eln=dict(component='StringEditQuantity',
                   props=dict(suggestions=['constant potential', 'pulsed potential', 'constant current']))
    )

    potential = Quantity(
        type=np.float64,
        unit=("mV"),
        a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='mV'))

    measurement_duration = Quantity(
        type=np.float64,
        unit=("s"),
        a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='s'))

    electrode_support = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    electrocatalyst_binder = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))

    cell_type = Quantity(
        type=str,
        a_eln=dict(component='StringEditQuantity', props=dict(
            suggestions=['H-type', 'flow', 'other'])))
    
    membrane = Quantity(
        type=str,
        a_eln=dict(component='StringEditQuantity', props=dict(
            suggestions=['other'])))

    counter_electrode = Quantity(type=str, a_eln=dict(component='StringEditQuantity', props=dict(
            suggestions=['Pt'])))
    reference_electrode = Quantity(type=str, a_eln=dict(component='StringEditQuantity', props=dict(
            suggestions=['RHE', 'Ag/AgCl'])))

    e_feed = SubSection(section_def=Feed)

    e_reaction_data = SubSection(section_def=ECatalyticReactionData)

    potentiostatic_measurement = SubSection(section_def=PotentiostaticMeasurement)
    measurement_details = SubSection(section_def=Measurement)

    def normalize(self, archive, logger):
        super(ElectroCatalyticReaction, self).normalize(archive, logger)

        if not self.data_file:
            return

        if os.path.splitext(
                self.data_file)[-1] != ".csv" and os.path.splitext(
                self.data_file)[-1] != ".xlsx":
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
        e_cat_data = ECatalyticReactionData()
        reactants = []
        reactant_names = []
        products = []
        product_names = []
        faradaic_efficiencies = []
        partial_current_densities = []
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
                reactant = Reactant(name=col_split[1],
                                    amount=data[col])
                reactant_names.append(col_split[1])
                reactants.append(reactant)

            if col_split[0] == "ToS":
                e_cat_data.time_on_stream = data[col]

            if col_split[0] == "gas_flow":
                feed.flow_rates = data[col]

            if col_split[0] == "partial current density":  # corresponds to reaction rate
                partial_current_densities = Ecat_Product(name=col_split[3],
                                                         partial_current_density=data[col])

                Ecat_Product.partial_current_density = data[col]
                partial_current_densities.append(partial_current_densities)

            if col_split[0] == "F.E.":  # Faradeic Efficiency (selectivity)
                product = Ecat_Product(name=col_split[1])
                for i, p in enumerate(products):
                    if p.name == col_split[1]:
                        product = products.pop(i)
                        break

                products.append(product)

                product.faradaic_efficiency = data[col]
                product_names.append(col_split[1])
                faradaic_efficiencies.append(data[col])

        for p in products:
            if p.selectivity is None or len(p.selectivity) == 0:
                p.selectivity = number_of_runs * [0]

        feed.reactants = reactants
        feed.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        e_cat_data.ecat_products = products
        e_cat_data.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        e_cat_data.product = faradaic_efficiencies

        self.feed = feed
        self.e_reaction_data = e_cat_data

        add_activity(archive)

        # if feed.reactants is not None:
        #     archive.results.properties.catalytic.reactivity.reactants = reactants
        #     archive.results.properties.catalytic.activity.conversion='cat_data.conversion/0:1/conversion_product_based'
        #     archive.results.properties.catalytic.reactivity.conversion_names = conversion_names
        # if products is not None:
        #     archive.results.properties.catalytic.reactivity.products = products
        if self.reaction_name:
            archive.results.properties.catalytic.reactivity.reaction_name = self.reaction_name
            archive.results.properties.catalytic.reactivity.reaction_class = self.reaction_class


m_package.__init_metainfo__()


m_package.__init_metainfo__()
