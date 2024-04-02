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

from nomad.metainfo import (Quantity, SubSection, Section, Datetime,)
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.basesections import (PubChemPureSubstanceSection)

from nomad.datamodel.results import (Results, Properties, CatalyticProperties, Reaction)

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
    if not archive.results.properties.catalytic.reaction:
        archive.results.properties.catalytic.reaction = Reaction()

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

    pure_component = SubSection(section_def=PubChemPureSubstanceSection)


    def normalize(self, archive, logger):
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
        
        if self.name in ['C5-1', 'C6-1', 'nC5', 'nC6','Unknown','inert']:
            return
        elif self.name == 'n-Butene':
            self.name = '1-butene'
        elif self.name == 'MAN':
            self.name = 'maleic anhydride'
        elif '_' in self.name:
            self.name = self.name.replace('_',' ')	
        # elif self.name == 'acetic_acid':
        #     self.name = 'acetic acid'
        if self.name and self.pure_component is None:
            self.pure_component = PubChemPureSubstanceSection(
                name=self.name
            )
            self.pure_component.normalize(archive, logger)
        
        if self.pure_component is not None and self.pure_component.iupac_name is None:
            if self.pure_component.molecular_formula == 'CO2':
                self.pure_component.iupac_name = 'carbon dioxide'

        if self.name == "CO" or self.name == "carbon monoxide":
            self.pure_component.iupac_name = 'carbon monoxide'
            self.pure_component.molecular_formula = 'CO'
            self.pure_component.molecular_mass = 28.01
            self.pure_component.smile = 'C#O'
            self.pure_component.inchi = 'InChI=1S/CO/c1-2'
            self.pure_component.inchi_key = 'UGFAIRIUMAVXCW-UHFFFAOYSA-N'
            self.pure_component.cas_number = '630-08-0'

        if self.name is None and self.pure_component is not None:
            self.name = self.pure_component.molecular_formula


class Reactant(Reagent):
    m_def = Section(label_quantity='name', description='A reagent that has a conversion in a reaction that is not null')

    gas_concentration_out = Quantity(
        type=np.float64, shape=['*'],
        description='Volumetric fraction of reactant in outlet.', 
        a_eln=ELNAnnotation(component='NumberEditQuantity'))
    
    reference = Quantity(type=Reagent, a_eln=dict(component='ReferenceEditQuantity'))
    conversion = Quantity(type=np.float64, shape=['*'])
    conversion_type = Quantity(type=str, a_eln=dict(component='StringEditQuantity', props=dict(
        suggestions=['product_based', 'reactant_based', 'unknown'])))
    conversion_product_based = Quantity(type=np.float64, shape=['*'])
    conversion_reactant_based = Quantity(type=np.float64, shape=['*'])


class CatalyticSectionConditions_static(ArchiveSection):
    m_def = Section(description='A class containing reaction conditions of a single run or set of conditions.')

    repeat_settings_for_next_run = Quantity(
        type=bool, a_eln=ELNAnnotation(component='BoolEditQuantity'))

    set_temperature = Quantity(
        type=np.float64, unit='K', a_eln=ELNAnnotation(component='NumberEditQuantity'))

    set_pressure = Quantity(
        type=np.float64, unit='bar', a_eln=ELNAnnotation(component='NumberEditQuantity', defaultDisplayUnit='bar'))

    set_total_flow_rate = Quantity(
        type=np.float64, unit='mL/minute', a_eln=ELNAnnotation(component='NumberEditQuantity'))

    duration = Quantity(
        type=np.float64, unit='hour', a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='hour'))

    weight_hourly_space_velocity = Quantity(
        type=np.float64, unit='mL/(g*hour)', a_eln=dict(component='NumberEditQuantity'))

    contact_time = Quantity(
        type=np.float64, unit='g*s/mL', a_eln=ELNAnnotation(label='W|F', component='NumberEditQuantity'))

    gas_hourly_space_velocity = Quantity(
        type=np.float64, unit='1/hour', a_eln=dict(component='NumberEditQuantity'))

    datetime = Quantity(
        type=Datetime,
        description='The date and time when this activity was started.',
        a_eln=ELNAnnotation(component='DateTimeEditQuantity', label='Starting Time'))

    time_on_stream = Quantity(
        type=np.float64, unit='hour', a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='hour'))

    description = Quantity(
        type=str, a_eln=dict(component='RichTextEditQuantity'))

    reagents = SubSection(section_def=Reagent, repeats=True)

    def normalize(self, archive, logger):
        super(CatalyticSectionConditions_static, self).normalize(archive, logger)

        for reagent in self.reagents:
            if reagent is None:
                raise ValueError('No reagents are defined')
            reagent.normalize(archive, logger)

        if self.set_total_flow_rate is None and self.reagents is not None:
            if self.reagents[0].flow_rate is not None:
                total_flow_rate=0
                for reagent in self.reagents:
                    if reagent.flow_rate is not None:
                        total_flow_rate+=reagent.flow_rate
                self.set_total_flow_rate=total_flow_rate

        if self.set_total_flow_rate is not None:
           for reagent in self.reagents:
                if reagent.flow_rate is None and reagent.gas_concentration_in is not None:
                    reagent.flow_rate = self.set_total_flow_rate * reagent.gas_concentration_in

        if self.weight_hourly_space_velocity is None and self.set_total_flow_rate is not None:
            try:
                self.weight_hourly_space_velocity = self.set_total_flow_rate / self.m_root().data.reactor_filling.catalyst_mass
            except:
                logger.warning('The catalyst mass is not defined. Needed to calculate the weight hourly space velocity.')
                return
        if self.contact_time is None and self.weight_hourly_space_velocity is not None:
            self.contact_time = 1 / self.weight_hourly_space_velocity
        
        if self.gas_hourly_space_velocity is None and self.set_total_flow_rate is not None:
            if self.m_root().data.reactor_filling.apparent_catalyst_volume is not None:
                self.gas_hourly_space_velocity = self.set_total_flow_rate / self.m_root().data.reactor_filling.apparent_catalyst_volume
            # except:
            #     logger.warning('The apparent catalyst volume is not defined. The gas hourly space velocity cannot be calculated.')
            #     return
            
        add_activity(archive)
        if self.set_temperature is not None:
            print(self.set_temperature)
            archive.results.properties.catalytic.reaction.temperature = self.set_temperature
        if self.set_pressure is not None:
            archive.results.properties.catalytic.reaction.pressure = self.set_pressure
        if self.set_total_flow_rate is not None:
            archive.results.properties.catalytic.reaction.flow_rate = self.set_total_flow_rate
        if self.weight_hourly_space_velocity is not None:
            archive.results.properties.catalytic.reaction.weight_hourly_space_velocity = self.weight_hourly_space_velocity

        if self.reagents is not None:
            archive.results.properties.catalytic.reaction.reactants = self.reagents


class CatalyticSectionConditions_dynamic(CatalyticSectionConditions_static):
    m_def = Section(description='A class containing reaction conditions of a generic reaction with changing conditions.')

    set_temperature = Quantity(
        type=np.float64, unit='K', a_eln=dict(label='Set temperature section start', component='NumberEditQuantity'))

    set_temperature_section_stop = Quantity(
        type=np.float64, unit='K', a_eln=dict(component='NumberEditQuantity'))

    set_pressure = Quantity(
        type=np.float64, unit='bar', a_eln=dict(label='Set pressure section start', component='NumberEditQuantity', defaultDisplayUnit='bar'))

    set_pressure_section_stop = Quantity(
        type=np.float64, unit='bar', a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='bar'))


class ReactionConditionsSimple(PlotSection, ArchiveSection):
    m_def = Section(description='A class containing reaction conditions for a generic reaction. with seperated runs containing the conditions')

    number_of_sections = Quantity(
        type=np.int32,
        description='The number of sections with different reaction conditions.',
        a_eln=dict(component='NumberEditQuantity'))

    total_time_on_stream = Quantity(
        type=np.float64, unit='hour', a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='hour'))

    section_runs = SubSection(section_def=CatalyticSectionConditions_static, repeats=True)

    def normalize(self, archive, logger):
        super(ReactionConditionsSimple, self).normalize(archive, logger)


        if self.section_runs is not None:
            for i,run in enumerate(self.section_runs):
                try:
                    if run.repeat_settings_for_next_run is True:
                        try:
                            if self.section_runs[i+1] is None:
                                self.section_runs.append(CatalyticSectionConditions_static())
                        except IndexError:
                            self.section_runs.append(CatalyticSectionConditions_static())
                        if run.set_temperature is not None and self.section_runs[i+1].set_temperature is None:
                            self.section_runs[i+1].set_temperature = run.set_temperature
                        if run.set_pressure is not None and self.section_runs[i+1].set_pressure is None:
                            self.section_runs[i+1].set_pressure = run.set_pressure
                        if run.set_total_flow_rate is not None and self.section_runs[i+1].set_total_flow_rate is None:
                            self.section_runs[i+1].set_total_flow_rate = run.set_total_flow_rate
                        if run.duration is not None and self.section_runs[i+1].duration is None:
                            self.section_runs[i+1].duration = run.duration
                        if run.weight_hourly_space_velocity is not None and self.section_runs[i+1].weight_hourly_space_velocity is None:
                            self.section_runs[i+1].weight_hourly_space_velocity = run.weight_hourly_space_velocity
                        if run.contact_time is not None and self.section_runs[i+1].contact_time is None:
                            self.section_runs[i+1].contact_time = run.contact_time
                        if run.gas_hourly_space_velocity is not None and self.section_runs[i+1].gas_hourly_space_velocity is None:
                            self.section_runs[i+1].gas_hourly_space_velocity = run.gas_hourly_space_velocity
                        if run.reagents is not None and self.section_runs[i+1].reagents == []:
                            reagents_next=[]
                            for reagent in run.reagents:
                                reagents_next.append(reagent)
                            self.section_runs[i+1].reagents = reagents_next
                except:
                    pass
            
            if self.section_runs[0].duration is not None:
                time=0
                for run in self.section_runs:
                    if run.duration is not None:
                        time = time + run.duration
                        run.time_on_stream = time
                self.total_time_on_stream = time

            self.number_of_sections = len(self.section_runs)


        add_activity(archive)
        for run in self.section_runs:
            run.normalize(archive, logger)

        #Figures definitions:
        self.figures = []

        if self.section_runs is not None:
            if len(self.section_runs) > 1:
                figT=go.Figure()
                x=[0,]
                y=[]
                for i,run in enumerate(self.section_runs):
                    if run.set_temperature is not None:
                        y.append(run.set_temperature.to('kelvin'))
                        try:
                            if run.set_temperature_section_stop is not None:
                                y.append(run.set_temperature_section_stop.to('kelvin'))
                        except:
                            y.append(run.set_temperature.to('kelvin'))
                    if run.set_pressure is not None:
                        if i == 0:
                            figP=go.Figure()
                            y_p=[]
                        y_p.append(run.set_pressure.to('bar'))
                        try:
                            if run.set_pressure_section_stop is not None:
                                y_p.append(run.set_pressure_section_stop.to('bar'))
                        except:
                            y_p.append(run.set_pressure.to('bar'))
                    if run.time_on_stream is not None:
                        x.append(run.time_on_stream.to('hour'))
                        if i != len(self.section_runs)-1:
                            x.append(run.time_on_stream.to('hour'))
                        x_text="time (h)"
                    elif i == len(self.section_runs)-1:
                        for j in range(1, len(self.section_runs)):
                            x.append(j)
                            if j != len(self.section_runs)-1:
                                x.append(j)
                        x_text='step'
                    if run.reagents != []:
                        for n,reagent in enumerate(run.reagents):
                            if n == 0 and i == 0:
                                figR=go.Figure()
                                reagent_n, runs_n = (len(run.reagents), len(self.section_runs))
                                y_r=[[0 for k in range(2*runs_n)] for l in range(reagent_n+1) ]
                                reagent_name=[0 for k in range(reagent_n+1)]
                            if i==0:
                                reagent_name[n]=reagent.name
                                if n == len(run.reagents)-1:
                                    reagent_name[n+1]=['total flow rate']
                            if reagent.flow_rate is not None:
                                if reagent.name == reagent_name[n]:
                                    y_r[n][2*i]=(reagent.flow_rate[0].to('mL/minute'))
                                    y_r[n][2*i+1]=(reagent.flow_rate[0].to('mL/minute'))
                                    y_r_text="Flow rates (mL/min)"
                                else:
                                    logger.warning('Reagent name has changed in run'+str(i+1)+'.')
                                    return
                                if n == len(run.reagents)-1:
                                    y_r[n+1][2*i]=(run.set_total_flow_rate.to('mL/minute'))
                                    y_r[n+1][2*i+1]=(run.set_total_flow_rate.to('mL/minute'))
                            elif reagent.gas_concentration_in is not None:
                                y_r[n][i]=(reagent.gas_concentration_in)
                                y_r_text="gas concentrations"
                            if i == len(self.section_runs)-1:
                                figR.add_trace(go.Scatter(x=x, y=y_r[n], name=reagent.name))
                                if n == len(run.reagents)-1:
                                    figR.add_trace(go.Scatter(x=x, y=y_r[n+1], name='Total Flow Rates'))
                figT.add_trace(go.Scatter(x=x, y=y, name='Temperature'))
                figT.update_layout(title_text="Temperature")
                figT.update_xaxes(title_text=x_text)
                figT.update_yaxes(title_text="Temperature (K)")
                self.figures.append(PlotlyFigure(label='Temperature', figure=figT.to_plotly_json()))

                try:
                    if figP is not None:
                        figP.add_trace(go.Scatter(x=x, y=y_p, name='Pressure'))
                        figP.update_layout(title_text="Pressure")
                        figP.update_xaxes(title_text=x_text,)
                        figP.update_yaxes(title_text="pressure (bar)")
                        self.figures.append(PlotlyFigure(label='Pressure', figure=figP.to_plotly_json()))
                except:
                    pass
                try:
                    if figR is not None:
                        figR.update_layout(title_text="Gas feed", showlegend=True)
                        figR.update_xaxes(title_text=x_text)
                        figR.update_yaxes(title_text=y_r_text)
                        self.figures.append(PlotlyFigure(label='Feed Gas', figure=figR.to_plotly_json()))
                except:
                    pass


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

    sampling_frequency = Quantity(  #maybe better use sampling interval?
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

        # if self.runs is None and self.set_temperature is not None:
        #     number_of_runs=len(self.set_temperature)
        #     self.runs= np.linspace(0, number_of_runs - 1, number_of_runs)
        # elif self.runs is None and self.set_pressure is not None:
        #     number_of_runs=len(self.set_pressure)
        # elif self.runs is None and self.set_total_flow_rate is not None:
        #     number_of_runs=len(self.set_total_flow_rate)
        # elif self.runs is None and self.gas_hourly_space_velocity is not None:
        #     number_of_runs=len(self.gas_hourly_space_velocity)
        # elif self.runs is None and self.weight_hourly_space_velocity is not None:
        #     number_of_runs=len(self.weight_hourly_space_velocity)
        # elif self.runs is None and self.time_on_stream is not None:
        #     number_of_runs=len(self.time_on_stream)
        # elif self.runs is None and self.reagents[0].gas_concentration_in is not None:
        #     number_of_runs=len(self.reagents[0].gas_concentration_in)
        # elif self.runs is not None:
        #     number_of_runs=len(self.runs)
        # else:
        #     number_of_runs=1
        

        # if self.set_pressure is not None:
        #     if len(self.set_pressure) == 1:
        #         for n in range(number_of_runs-1):
        #             self.set_pressure=np.append(self.set_pressure, self.set_pressure[0])

        # if self.set_total_flow_rate is None and self.reagents is not None:
        #     self.set_total_flow_rate = np.array([])
        #     for n in range(number_of_runs):
        #         total_flow_rate=0
        #         for reagent in self.reagents:
        #             if reagent.flow_rate is not None:
        #                 if len(reagent.flow_rate) == 1:
        #                     for m in range(number_of_runs-1):
        #                         reagent.flow_rate=np.append(reagent.flow_rate, reagent.flow_rate[0])
        #                 elif len(reagent.flow_rate) != number_of_runs:
        #                     raise ValueError('The number of flow rates is not equal to the number of runs')
        #                 total_flow_rate+=reagent.flow_rate[n]
        #         self.set_total_flow_rate=np.append(self.set_total_flow_rate, total_flow_rate)

        # if self.set_total_flow_rate is not None:
        #     if len(self.set_total_flow_rate) == 1:
        #         set_total_flow_rate=[]
        #         for n in range(number_of_runs):
        #             set_total_flow_rate.append(self.set_total_flow_rate)
        #         self.set_total_flow_rate=set_total_flow_rate

        # for reagent in self.reagents:
        #     if reagent.gas_concentration_in is not None:
        #         if len(reagent.gas_concentration_in) == 1:
        #             gas_concentration_in=[]
        #             for n in range(number_of_runs):
        #                 gas_concentration_in.append(reagent.gas_concentration_in)
        #             reagent.gas_concentration_in=gas_concentration_in
        #         elif len(reagent.gas_concentration_in) != number_of_runs:
        #             print(len(reagent.gas_concentration_in), number_of_runs)
        #             raise ValueError('The number of gas concentrations is not equal to the number of runs')
            
        #     if reagent.flow_rate is not None:
        #         if len(reagent.flow_rate) == 1:
        #             for n in range(number_of_runs):
        #                 reagent.flow_rate.append(reagent.flow_rate[0])
        #         elif len(reagent.flow_rate) != number_of_runs:
        #             raise ValueError('The number of flow rates is not equal to the number of runs')

        # add_activity(archive)

        # if self.set_temperature is not None:
        #     archive.results.properties.catalytic.reaction.temperature = self.set_temperature
        # if self.set_pressure is not None:
        #     archive.results.properties.catalytic.reaction.pressure = self.set_pressure
        # if self.set_total_flow_rate is not None:
        #     archive.results.properties.catalytic.reaction.flow_rate = self.set_total_flow_rate
        # if self.weight_hourly_space_velocity is not None:
        #     archive.results.properties.catalytic.reaction.weight_hourly_space_velocity = self.weight_hourly_space_velocity
        # if self.reagents is not None:
        #     archive.results.properties.catalytic.reaction.reactants = self.reagents

        #Figures definitions for ReactionConditions Subsection:
        if self.time_on_stream is not None:
            x=self.time_on_stream.to('hour')
            x_text="time (h)"
        elif self.runs is not None:
            x=self.runs
            x_text="steps"
        else:
            return

        if self.set_temperature is not None and len(self.set_temperature) > 1:
            figT = px.scatter(x=x, y=self.set_temperature.to('kelvin'))
            figT.update_layout(title_text="Temperature")
            figT.update_xaxes(title_text=x_text,)
            figT.update_yaxes(title_text="Temperature (K)")
            self.figures.append(PlotlyFigure(label='Temperature', figure=figT.to_plotly_json()))

        if self.set_pressure is not None and len(self.set_pressure) > 1:
            figP = px.scatter(x=x, y=self.set_pressure.to('bar'))
            figP.update_layout(title_text="Pressure")
            figP.update_xaxes(title_text=x_text,)
            figP.update_yaxes(title_text="pressure (bar)")
            self.figures.append(PlotlyFigure(label='Pressure', figure=figP.to_plotly_json()))
        
        if self.reagents is not None and self.reagents != []:
            if self.reagents[0].flow_rate is not None or self.reagents[0].gas_concentration_in is not None: 
                fig5 = go.Figure()
                for i,r in enumerate(self.reagents):
                    if r.flow_rate is not None:
                        y=r.flow_rate.to('mL/minute')
                        fig5.add_trace(go.Scatter(x=x, y=y, name=r.name))
                        y5_text="Flow rates (mL/min)"
                        if self.set_total_flow_rate is not None and i == 0:
                            fig5.add_trace(go.Scatter(x=x,y=self.set_total_flow_rate, name='Total Flow Rates'))
                    elif self.reagents[0].gas_concentration_in is not None:
                        fig5.add_trace(go.Scatter(x=x, y=self.reagents[i].gas_concentration_in, name=self.reagents[i].name))    
                        y5_text="gas concentrations"
                fig5.update_layout(title_text="Gas feed", showlegend=True)
                fig5.update_xaxes(title_text=x_text)
                fig5.update_yaxes(title_text=y5_text)
                self.figures.append(PlotlyFigure(label='Feed Gas', figure=fig5.to_plotly_json()))


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


class Product(Reagent, ArchiveSection):
    m_def = Section(label_quantity='name', description='a chemical substance formed in the reaction mixture during a reaction')

    gas_concentration_out = Quantity(
        type=np.float64, shape=['*'],
        description='Volumetric fraction of reactant in outlet.', 
        a_eln=ELNAnnotation(component='NumberEditQuantity'))

    selectivity = Quantity(
        type=np.float64, shape=['*'],
        description='The selectivity of the product in the reaction mixture.',
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
        iris=['https://w3id.org/nfdi4cat/voc4cat_0000125'])
    
    product_yield = Quantity(type=np.float64, shape=['*'],
                             description='The yield of the product in the reaction mixture, calculated as conversion x selectivity.',
                             a_eln=ELNAnnotation(component='NumberEditQuantity'))
    
    rates = SubSection(section_def=Rates)
    
    def normalize(self, archive, logger):
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


class ReactorSetup(ArchiveSection):
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

    reactants_conversions = SubSection(section_def=Reactant, repeats=True)
    rates = SubSection(section_def=Rates, repeats=True)

    products = SubSection(section_def=Product, repeats=True)

    def normalize(self, archive, logger):
        
        if self.products is not None:
            for product in self.products:
                product.normalize(archive, logger)

class CatalyticReactionData(PlotSection, CatalyticReactionData_core, ArchiveSection):

    c_balance = Quantity(
        type=np.dtype(
            np.float64), shape=['*'])

