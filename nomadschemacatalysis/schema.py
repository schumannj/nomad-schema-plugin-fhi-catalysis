import numpy as np
import os

from ase.data import chemical_symbols

from nomad.metainfo import (
    Quantity,
    Section,
    SubSection,
    Package)

from nomad.datamodel.metainfo.eln import (
    System, Ensemble, Substance,
    Measurement)

# from nomad.datamodel.metainfo.basesections import SubstanceComponent

from nomad.datamodel.data import ArchiveSection

from nomad.datamodel.results import (Results, Material, Properties, HeterogeneousCatalysis,
                                     CatalystCharacterization, Reactivity)
from nomad.datamodel.data import EntryData, UseCaseElnCategory

from .catalytic_measurement import (
    CatalyticReactionData, Feed, Reagent, Conversion, Rates, Reactor_setup,
    )

from nomad.datamodel.results import Product, Reactant

m_package = Package(name='catalysis')


def add_catalyst(archive):
    '''Adds metainfo structure for catalysis data.'''
    if not archive.results:
        archive.results = Results()
    if not archive.results.properties:
        archive.results.properties = Properties()
    if not archive.results.properties.catalytic:
        archive.results.properties.catalytic = HeterogeneousCatalysis()
    if not archive.results.properties.catalytic.catalyst_characterization:
        archive.results.properties.catalytic.catalyst_characterization = CatalystCharacterization()


def add_activity(archive):
    '''Adds metainfo structure for catalysis activity test data.'''
    if not archive.results:
        archive.results = Results()
    if not archive.results.properties:
        archive.results.properties = Properties()
    if not archive.results.properties.catalytic:
        archive.results.properties.catalytic = HeterogeneousCatalysis()
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
        person or persons preparing the sample in the lab
        """,
        a_eln=dict(component='EnumEditQuantity'),
        repeats=True
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
        type=np.dtype(np.float64),
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
        a_eln=dict(hide=['cas_uri', 'cas_number', 'cas_name', 'inchi', 'inchi_key',
                         'smile', 'canonical_smile', 'cas_synonyms', 'molecular mass']),
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
            archive.results.properties.catalytic.catalyst_characterization.catalyst_type = self.catalyst_type
        if self.surface_area is not None:
            archive.results.properties.catalytic.catalyst_characterization.surface_area = self.surface_area.surfacearea
            archive.results.properties.catalytic.catalyst_characterization.method_surface_area = self.surface_area.method_surface_area_determination
        if self.preparation_details is not None:
            archive.results.properties.catalytic.catalyst_characterization.preparation_method = self.preparation_details.preparation_method


class CatalyticReaction(EntryData):

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
            'x': 'runs',
            'y': ['reaction_data/rates/:/reaction_rate'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}}]
        # {
        #     "label": "Conversion X [%]",
        #     'x': 'runs',
        #     'y': ['reaction_data/conversion/:/conversion_product_based'],
        #     'layout': {"showlegend": True,
        #                'yaxis': {
        #                    "fixedrange": False}, 'xaxis': {
        #                    "fixedrange": False}}, "config": {
        #         "editable": True, "scrollZoom": True}},
        # ]
    )

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

    data_file = Quantity(
        type=str,
        description="""
        excel or csv file that contains results of a catalytic measurement with
        temperature, (pressure,) gas feed composition, yield, rates and selectivities
        """,
        a_eln=dict(component='FileEditQuantity'),
        a_browser=dict(adaptor='RawFileAdaptor'))

    # data_file_h5 = Quantity(
    #     type=str,
    #     description="""
    #     excel or csv file that contains results of a catalytic measurement with
    #     temperature, (pressure,) gas feed composition, yield, rates and selectivities
    #     """,
    #     a_eln=dict(component='FileEditQuantity'),
    #     a_browser=dict(adaptor='RawFileAdaptor')
    # )

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

        # if (self.data_file_h5 is not None) and (os.path.splitext(
        #         self.data_file)[-1] != ".h5"):
        #     raise ValueError("Unsupported file format. This should be a hdf5 file ending with '.h5'" )
        #     return

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
                reagent = Reagent(name=col_split[1], gas_fraction=data[col])
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
                feed.flow_rates = np.nan_to_num(data[col])

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
                conversion2 = Reactant(name=col_split[1], conversion=np.nan_to_num(data[col]))

                conversion = Conversion(name=col_split[1], conversion=np.nan_to_num(data[col]), type='reactant-based conversion', conversion_reactant_based = np.nan_to_num(data[col]))
                for i, p in enumerate(conversions):
                    if p.name == col_split[1]:
                        conversion = conversions.pop(i)
                conversion.conversion_reactant_based = data[col]
                conversions.append(conversion)

                conversions2.append(conversion2)

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
        if products is not None:
            archive.results.properties.catalytic.reactivity.products = products
        if self.reaction_name is not None:
            archive.results.properties.catalytic.reactivity.reaction_name = self.reaction_name
            archive.results.properties.catalytic.reactivity.reaction_class = self.reaction_class

        if self.sample_reference is not None:
            if not archive.results.properties.catalytic.catalyst_characterization:
                archive.results.properties.catalytic.catalyst_characterization = CatalystCharacterization()

            if self.sample_reference.catalyst_type is not None:
                archive.results.properties.catalytic.catalyst_characterization.catalyst_type = self.sample_reference.catalyst_type
            if self.sample_reference.preparation_details is not None:
                archive.results.properties.catalytic.catalyst_characterization.preparation_method = self.sample_reference.preparation_details.preparation_method
            if self.sample_reference.surface_area is not None:
                archive.results.properties.catalytic.catalyst_characterization.surface_area = self.sample_reference.surface_area.surfacearea

        if self.sample_reference.elemental_composition is not None:
            if not archive.results:
                archive.results = Results()
            if not archive.results.material:
                archive.results.material = Material()

            try:
                archive.results.material.elemental_composition = self.sample_reference.elemental_composition
                if self.sample_reference.elemental_composition.element not in chemical_symbols:
                    logger.warn(
                        f"'{self.sample_reference.elemental_composition.element}' is not a valid element symbol and this "
                        "elemental_composition section will be ignored.")
                elif self.sample_reference.elemental_composition.element not in archive.results.material.elements:
                    archive.results.material.elements += [self.sample_reference.elemental_composition.element]

            except Exception as e:
                logger.warn('Could not analyse elemental compostion.', exc_info=e)


m_package.__init_metainfo__()
