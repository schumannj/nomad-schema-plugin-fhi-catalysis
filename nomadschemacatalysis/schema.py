from nomad.metainfo import Quantity, Package
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum

import numpy as np
import os
from nomad.units import ureg

from nomad.metainfo import (
    Quantity,
    Section,
    SectionProxy,
    SubSection,
    MSection, Package, Datetime)

from nomad.datamodel.metainfo.eln import (
    Substance,
    Measurement,
    ElnWithFormulaBaseSection)

from nomad.datamodel.results import (Results, Material, Properties, HeterogeneousCatalysis,
                                     CatalystCharacterization, Reactivity)
from nomad.datamodel.data import EntryData, UseCaseElnCategory

from .catalytic_measurement import (
    CatalyticReactionData, Feed, Product, Reactant, Conversion, Rate)

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


class CatalystSample(Substance, EntryData):
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

    surface_area = Quantity(
        type=np.dtype(np.float64),
        unit=("m**2/g"),
        a_eln=dict(
            component='NumberEditQuantity', defaultDisplayUnit='m**2/g',
        ))

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
        a_eln=dict(component='EnumEditQuantity')
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

    storing_institution = Quantity(
        type=str,
        shape=[],
        description="""
        institution at which the sample is stored
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(
            suggestions=['Fritz-Haber-Institut Berlin / Abteilung AC',
                         'Fritz-Haber-Institut Berlin / ISC']))
        )

    catalyst_type = Quantity(
        type=str,
        shape=[],
        description="""
          classification of catalyst type
          """,
        a_eln=dict(
            component='EnumEditQuantity', props=dict(
                suggestions=['bulk catalyst', 'supported catalyst', 'model catalyst',
                             'layered catalyst', 'other', 'unkown'])))

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
        logger.info('CatalystSampleSection.normalize called')

        add_catalyst(archive)

        if self.surface_area is not None:
            archive.results.properties.catalytic.catalyst_characterization.surface_area = self.surface_area
        if self.catalyst_type is not None:
            archive.results.properties.catalytic.catalyst_characterization.catalyst_type = self.catalyst_type
        if self.preparation_method is not None:
            archive.results.properties.catalytic.catalyst_characterization.preparation_method = self.preparation_method


class CatalyticReaction(Measurement, EntryData):  # used to be MSection
    """
    This schema is adapted to map the data of the clean Oxidation dataset (JACS,
    https://doi.org/10.1021/jacs.2c11117) The descriptions in the quantities
    represent the instructions given to the user who manually curated the data.
    """

    m_def = Section(
        label='Heterogeneous Catalysis - Activity Test',
        categories=[UseCaseElnCategory],
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
                'Oxidation of Butane', 'Methanol Synthesis', 'Fischer-Tropsch',
                'Water gas shift reaction', 'Ammonia Synthesis', 'Ammonia decomposition'])))

    institute = Quantity(
        type=str,
        shape=[],
        description="""
        institution at which the measurement was performed
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(
            suggestions=['Fritz-Haber-Institut Berlin / Abteilung AC',
                         'Fritz-Haber-Institut Berlin / ISC',
                         'TU Berlin, BASCat', 'HZB','CATLAB']))
    )

    experimentator = Quantity(
        type=str,
        shape=[],
        description="""
        person that performed or started the measurement
        """,
        a_eln=dict(component='EnumEditQuantity')
    )

    mass = Quantity(
        type=np.dtype(np.float64),
        unit=("mg"),
        a_eln=dict(
            component='NumberEditQuantity', defaultDisplayUnit='mg'))

    data_file = Quantity(
        type=str,
        a_eln=dict(component='FileEditQuantity'),
        a_browser=dict(adaptor='RawFileAdaptor'))

    feed = SubSection(section_def=Feed)
    data = SubSection(section_def=CatalyticReactionData)

    measurement_details = SubSection(section_def=Measurement)

    def normalize(self, archive, logger):
        super(CatalyticReaction, self).normalize(archive, logger)
        logger.info('CatalyticReaction.normalize called')

        if not self.data_file or (os.path.splitext(
                self.data_file)[-1] != ".csv" and os.path.splitext(
                self.data_file)[-1] != ".xlsx"):
            raise ValueError("Unsupported file format. Only xlsx and .csv files")
            return

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
        reactants = []
        reactant_names = []
        products = []
        product_names = []
        selectivities = []
        conversions = []
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
                reactant = Reactant(name=col_split[1],
                                    amount=data[col])
                reactant_names.append(col_split[1])
                reactants.append(reactant)
            if col_split[0] == "temperature":
                cat_data.temperature = data[col]
                cat_data.temperature_max = max(data[col])
                cat_data.temperature_min = min(data[col])

            if col_split[0] == "time":
                cat_data.time_on_stream = data[col]

            if col_split[0] == "C-balance":
                cat_data.c_balance = data[col]

            if col_split[0] == "GHSV":
                feed.space_velocity = data[col]

            if col_split[0] == "r":  # reaction rate
                rate = Rate(name=col_split[1],
                            reaction_rate=data[col])

                rate.reaction_rate = data[col]
                rates.append(rate)

            if len(col_split) < 3 or col_split[2] != '(%)':
                continue

            if col_split[0] == "x_p":  # conversion, based on product detection
                conversion = Conversion(name=col_split[1],
                                        conversion_product_based=data[col])
                for i, p in enumerate(conversions):
                    if p.name == col_split[1]:
                        conversion = conversions.pop(i)

                conversion.conversion_product_based = data[col]

                conversion_names.append(col_split[1])
                conversion_list.append(data[col])
                conversions.append(conversion)

            if col_split[0] == "x_r":  # conversion, based on reactant detection
                conversion = Conversion(name=col_split[1])
                for i, p in enumerate(conversions):
                    if p.name == col_split[1]:
                        conversion = conversions.pop(i)

                conversion.conversion_reactant_based = data[col]
                conversions.append(conversion)

            if col_split[0] == "S_p":  # selectivity
                product = Product(name=col_split[1])
                for i, p in enumerate(products):
                    if p.name == col_split[1]:
                        product = products.pop(i)
                        break

                products.append(product)

                product.selectivity = data[col]
                product_names.append(col_split[1])
                selectivities.append(data[col])

        for p in products:
            if p.selectivity is None or len(p.selectivity) == 0:
                p.selectivity = number_of_runs * [0]

        for c in conversions:
            if c.conversion_product_based is None or len(c.conversion_product_based) == 0:
                c.conversion_product_based = number_of_runs * [0]

        feed.reactants = reactants
        feed.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        cat_data.products = products
        cat_data.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        cat_data.conversion = conversions
        cat_data.rates = rates

        self.feed = feed
        self.data = cat_data

        add_activity(archive)

        if feed.reactants is not None:
            archive.results.properties.catalytic.reactivity.reactants = reactant_names
            archive.results.properties.catalytic.reactivity.test_temperature_high = cat_data.temperature_max
            archive.results.properties.catalytic.reactivity.test_temperature_low = cat_data.temperature_min
            archive.results.properties.catalytic.reactivity.test_temperatures = cat_data.temperature
            # archive.results.properties.catalytic.activity.conversion='cat_data.conversion/0:1/conversion_product_based'
            archive.results.properties.catalytic.reactivity.conversion_names = conversion_names
        if products is not None:
            archive.results.properties.catalytic.reactivity.products = product_names
            # archive.results.properties.catalytic.activity.selectivity=selectivities
        if self.reaction_name:
            archive.results.properties.catalytic.reactivity.reaction_name = self.reaction_name
            archive.results.properties.catalytic.reactivity.reaction_class = self.reaction_class

        if self.sample_reference is not None:
            if not archive.results.properties.catalytic.catalyst_characterization:
                archive.results.properties.catalytic.catalyst_characterization = CatalystCharacterization()

            if self.sample_reference.surface_area is not None:
                archive.results.properties.catalytic.catalyst_characterization.surface_area = self.sample_reference.surface_area
            if self.sample_reference.catalyst_type is not None:
                archive.results.properties.catalytic.catalyst_characterization.catalyst_type = self.sample_reference.catalyst_type
            if self.sample_reference.preparation_method is not None:
                archive.results.properties.catalytic.catalyst_characterization.preparation_method = self.sample_reference.preparation_method

        if self.sample_reference.molecular_formula is not None:
            if not archive.results:
                archive.results = Results()
            if not archive.results.material:
                archive.results.material = Material()

            try:
                from nomad.atomutils import Formula
                formula = Formula(self.sample_reference.molecular_formula)
                formula.populate(archive.results.material)
                # if not self.sample_reference.elemental_composition:
                #     mass_fractions = formula.mass_fractions()
                #     for element, fraction in formula.atomic_fractions().items():
                #         self.sample_reference.elemental_composition.append(
                #             ElementalComposition(
                #                 element=element,
                #                 atomic_fraction=fraction,
                #                 mass_fraction=mass_fractions[element],
                #             )
                #         )
            except Exception as e:
                logger.warn('Could not analyse chemical formula.', exc_info=e)



m_package.__init_metainfo__()