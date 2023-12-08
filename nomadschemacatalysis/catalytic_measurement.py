#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np

from nomad.metainfo import (Quantity, SubSection, Section)
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.basesections import (PubChemPureSubstanceSection, PureSubstanceComponent)

from nomad.datamodel.results import (Results, Properties, CatalyticProperties, Reactivity)

from nomad.datamodel.metainfo.plot import PlotSection, PlotlyFigure
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json

from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
)

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

class Reagent(ArchiveSection):
    m_def = Section(label_quantity='name', description='a chemical substance present in the initial reaction mixture')
    name = Quantity(type=str, a_eln=ELNAnnotation(label='reagent name', component='StringEditQuantity'), description="reagent name")
    gas_concentration_in = Quantity(
        type=np.float64, shape=['*'],
        description='Volumetric fraction of reactant in feed.', 
        a_eln=ELNAnnotation(component='NumberEditQuantity'))
    flow_rate = Quantity(
        type=np.float64, shape=['*'], unit='mL/minutes',
        description='Flow rate of reactant in feed.',
        a_eln=ELNAnnotation(component='NumberEditQuantity'))

    pure_reagent = SubSection(section_def=PubChemPureSubstanceSection)


    def normalize(self, archive, logger: 'BoundLogger') -> None:
        '''
        The normalizer for the `PureSubstanceComponent` class. If none is set, the
        normalizer will set the name of the component to be the molecular formula of the
        substance.

        Args:
            archive (EntryArchive): The archive containing the section that is being
            normalized.
            logger ('BoundLogger'): A structlog logger.
        '''
        super(Reagent, self).normalize(archive, logger)
        
        if self.name and self.pure_reagent is None:
            self.pure_reagent = PubChemPureSubstanceSection(
                name=self.name
            )
            self.pure_reagent.normalize(archive, logger)
        
        if self.name is None and self.pure_reagent is not None:
            self.name = self.pure_reagent.molecular_formula
        

class Conversion(ArchiveSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    reference = Quantity(type=Reagent, a_eln=dict(component='ReferenceEditQuantity'))
    type = Quantity(type=str, a_eln=dict(component='StringEditQuantity', props=dict(
        suggestions=['product_based', 'reactant_based', 'unknown'])))
    conversion = Quantity(type=np.float64, shape=['*'])
    conversion_product_based = Quantity(type=np.float64, shape=['*'])
    conversion_reactant_based = Quantity(type=np.float64, shape=['*'])


class Reactant(Reagent):
    m_def = Section(label_quantity='name', description='A reagent that has a conversion in a reaction that is not null')

    conversion = SubSection(section_def=Conversion)

class ReactionConditions(PlotSection, ArchiveSection):
    m_def = Section(description='A class containing reaction conditions for a generic reaction.')

    set_temperature = Quantity(
        type=np.float64, shape=['*'], unit='K', a_eln=ELNAnnotation(component='NumberEditQuantity'))
    
    set_pressure = Quantity(
        type=np.float64, shape=['*'], unit='bar', a_eln=ELNAnnotation(component='NumberEditQuantity', defaultDisplayUnit='bar'))
    
    set_total_flow_rate = Quantity(
        type=np.float64, shape=['*'], unit='mL/minute', a_eln=ELNAnnotation(component='NumberEditQuantity'))
    
    weight_hourly_space_velocity = Quantity(
        type=np.float64, shape=['*'], unit='mL/(g*hour)', a_eln=dict(component='NumberEditQuantity'))

    contact_time = Quantity(
        type=np.float64, shape=['*'], unit='g*s/mL', a_eln=ELNAnnotation(label='W|F'))

    gas_hourly_space_velocity = Quantity(
        type=np.float64, shape=['*'], unit='1/hour', a_eln=dict(component='NumberEditQuantity'))

    runs = Quantity(type=np.float64, shape=['*'])

    sampling_frequency = Quantity(
        type=np.float64, shape=[], unit='Hz',
        description='The number of measurement points per time.',
        a_eln=dict(component='NumberEditQuantity'))
    
    time_on_stream = Quantity(
        type=np.float64, shape=['*'], unit='hour', a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='hour'))
    
    reagents = SubSection(section_def=Reagent, repeats=True)

    def normalize(self, archive, logger):
        super(ReactionConditions, self).normalize(archive, logger)
        for reagent in self.reagents:
            reagent.normalize(archive, logger)
        
        if self.runs is None and self.set_temperature is not None:
            number_of_runs=len(self.set_temperature)
            self.runs= np.linspace(0, number_of_runs - 1, number_of_runs)
        else:
            number_of_runs=len(self.runs)
        
        if self.set_pressure is not None:
            if len(self.set_pressure) == 1:
                set_pressure=[]
                for n in range(number_of_runs):
                    set_pressure.append(self.set_pressure)
                self.set_pressure=set_pressure

        if self.set_total_flow_rate is not None:
            if len(self.set_total_flow_rate) == 1:
                set_total_flow_rate=[]
                for n in range(number_of_runs):
                    set_total_flow_rate.append(self.set_total_flow_rate)
                self.set_total_flow_rate=set_total_flow_rate


        add_activity(archive)

        if self.set_temperature is not None:
            archive.results.properties.catalytic.reactivity.test_temperatures = self.set_temperature
        if self.set_pressure is not None:
            archive.results.properties.catalytic.reactivity.pressure = self.set_pressure
        if self.set_total_flow_rate is not None:
            archive.results.properties.catalytic.reactivity.flow_rate = self.set_total_flow_rate
        if self.weight_hourly_space_velocity is not None:
            archive.results.properties.catalytic.reactivity.weight_hourly_space_velocity = self.weight_hourly_space_velocity
        if self.reagents is not None:
            archive.results.properties.catalytic.reactivity.reactants = self.reagents
            #archive.results.properties.catalytic.reactivity.reactants.gas_consentration_in = self.reagents.gas_concentration_in
    
        #Figures definitions:
        if self.time_on_stream is not None:
            x=self.time_on_stream.to('hour')
            x_text="time (h)"
        else:
            x=self.runs
            x_text="steps" 

        if self.set_temperature is not None:
            figT = px.scatter(x=x, y=self.set_temperature.to('kelvin'))
            figT.update_layout(title_text="Temperature")
            figT.update_xaxes(title_text=x_text,) 
            figT.update_yaxes(title_text="Temperature (K)")
            self.figures.append(PlotlyFigure(label='Temperature', figure=figT.to_plotly_json()))
        
        if self.reagents is not None and (self.reagents[0].flow_rate is not None or self.reagents[0].gas_concentration_in is not None):
            fig5 = go.Figure()
            for i,c in enumerate(self.reagents):
                if self.reagents[0].flow_rate is not None:
                    fig5.add_trace(go.Scatter(x=x, y=self.reagents[i].flow_rate, name=self.reaction_conditions.reagents[i].name))
                    y5_text="Flow rates ()"
                    if self.set_total_flow_rate is not None and i == 0:
                        fig5.add_trace(go.Scatter(x=x,y=self.set_total_flow_rate, name='Total Flow Rates'))
                elif self.reagents[0].gas_concentration_in is not None:
                    fig5.add_trace(go.Scatter(x=x, y=self.reagents[i].gas_concentration_in, name=self.reagents[i].name))    
                    y5_text="gas concentrations"
            fig5.update_layout(title_text="Gas feed", showlegend=True)
            fig5.update_xaxes(title_text=x_text) 
            fig5.update_yaxes(title_text=y5_text)
            self.figures.append(PlotlyFigure(label='Feed Gas', figure=fig5.to_plotly_json()))



class Feed(ReactionConditions, ArchiveSection):  


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
    diluent_sievefraction_high = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))
    diluent_sievefraction_low = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))
    catalyst_mass = Quantity(
        type=np.float64, shape=[], unit='g',
        a_eln=dict(component='NumberEditQuantity'))
    apparent_catalyst_volume = Quantity(
        type=np.float64, shape=[], unit='mL', a_eln=ELNAnnotation(component='NumberEditQuantity'))
    catalyst_sievefraction_high = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))
    catalyst_sievefraction_low = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))
    particle_size = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))
    
    def normalize(self, archive, logger):
        super(Feed, self).normalize(archive, logger)

class Rates(ArchiveSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, a_eln=ELNAnnotation(component='StringEditQuantity'))

    reaction_rate = Quantity(
        type=np.float64,
        shape=['*'],
        unit='mmol/g/hour',
        description='The reaction rate for mmol of product (or reactant) formed (depleted) per catalyst (g) per time (hour).')
    specific_mass_rate = Quantity(
        type=np.float64, shape=['*'], unit='mmol/g/hour',
        description='The specific reaction rate normalized by active (metal) catalyst mass, instead of mass of total catalyst.')
    specific_surface_area_rate = Quantity(
        type=np.float64, shape=['*'], unit='mmol/m**2/hour',
        description='The specific reaction rate normalized by active (metal) surface area of catalyst, instead of mass of total catalyst.')
    space_time_yield = Quantity(
        type=np.float64,
        shape=['*'],
        unit='g/g/hour',
        description='The amount of product formed (in g), per total catalyst (g) per time (hour).')
    rate = Quantity(
        type=np.float64, shape=['*'], unit='g/g/hour',
        description='The amount of reactant converted (in g), per total catalyst (g) per time (hour).'
    )
    turn_over_frequency = Quantity(
        type=np.float64, shape=['*'], unit='1/hour',
        description='The turn oder frequency, calculated from mol of reactant or product, per number of sites, over time.')


class Product(Rates, ArchiveSection):
    m_def = Section(label_quantity='name')

    selectivity = Quantity(type=np.float64, shape=['*'])
    product_yield = Quantity(type=np.float64, shape=['*'])

    pure_product = SubSection(section_def=PubChemPureSubstanceSection)

    def normalize(self, archive, logger: 'BoundLogger') -> None:
        '''
        The normalizer for the adjusted `PureSubstanceComponent` class. If none is set, the
        normalizer will set the name of the component to be the molecular formula of the
        substance.

        Args:
            archive (EntryArchive): The archive containing the section that is being
            normalized.
            logger ('BoundLogger'): A structlog logger.
        '''
        super(Product, self).normalize(archive, logger)
        
        if self.name and self.pure_product is None:
            self.pure_product = PubChemPureSubstanceSection(
                name=self.name
            )
            self.pure_product.normalize(archive, logger)
        
        if self.name is None and self.pure_product is not None:
            self.name = self.pure_product.molecular_formula

class Reactor_setup(ArchiveSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, shape=[], a_eln=dict(component='EnumEditQuantity'))
    reactor_type = Quantity(type=str, shape=[], a_eln=dict(component='EnumEditQuantity'),
                             props=dict(suggestions=['plug flow reactor', 'batch reactor', 'continuous stirred-tank reactor', 'fluidized bed']))
    bed_length = Quantity(type=np.float64, shape=[], unit='mm',
                          a_eln=dict(component='NumberEditQuantity'))
    reactor_cross_section_area = Quantity(type=np.float64, shape=[], unit='mm**2',
                                          a_eln=dict(component='NumberEditQuantity'))
    reactor_diameter = Quantity(type=np.float64, shape=[], unit='mm',
                                          a_eln=dict(component='NumberEditQuantity'))
    reactor_volume = Quantity(type=np.float64, shape=[], unit='ml',
                              a_eln=dict(component='NumberEditQuantity'))
    

class CatalyticReactionData_core(ArchiveSection):
    temperature = Quantity(
        type=np.float64, shape=['*'], unit='°C')

    pressure = Quantity(
        type=np.float64, shape=['*'], unit='bar')

    runs = Quantity(type=np.float64, shape=['*'])
    time_on_stream = Quantity(type=np.float64, shape=['*'], unit='hour')

    reactants_conversions = SubSection(section_def=Conversion, repeats=True)
    rates = SubSection(section_def=Rates, repeats=True)


class CatalyticReactionData(PlotSection, CatalyticReactionData_core, ArchiveSection):

    c_balance = Quantity(
        type=np.dtype(
            np.float64), shape=['*'])
    products = SubSection(section_def=Product, repeats=True)

    def normalize(self, archive, logger):
        for product in self.products:
            product.normalize(archive, logger)
