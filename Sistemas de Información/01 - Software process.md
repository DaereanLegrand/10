# Software process description
At describing processes, we talk about specifying a data model, designing a user interface, and the ordering of these activities. 

Including:
- Products, which are the outcomes of a process activity.
- Roles, reflecting responsibilities of the people involved.
- Post-conditions, statements that are true before and after the act is enacted or produced.

Minimum being the standardization of the process.

Plan-driven process, when is planned in advance and progress is measured against this plan.

Agile methods, planning is incremental and easier to change to the ever changing user requirements. In many cases both are used at the same time, with no good or wrong choice.

# Software process model
Most large systems are developed using a process that incorporates elements from all of these model.
## Waterfall model
Different stages, between specification and development.
### Benefits
- Process is visible.
### Problems
- Not likely to changes, yet hard to respond to them.
- Used mostly for large systems.
- Not a lot of business systems need them.
## Incremental model
Specification, development and validation are interleaved. May be plan-driven or agile.
### Benefits
- Costs are reduced per change.
- Easier to get feedback of the work that has been done.
- Faster delivery and development.
### Problems
- Process is not visible.
- Tends to degrade as new components are added.
## Integration and configuration
The system is assembled from existing configurable process.
### Types of reusable software
COTS configured for a service in particular, developed as a package or as a framework, may be available for remote invocation.
### Benefits
- Reduced costs.
- Faster delivery and deployment of system.
Problems
- Requirement components are inevitable.
- Loss of control over evolution.

# Process activities
Real software are inter-leaved sequences of technical, collaborative and managerial activities with the goal of specifying, designing, implementing and testing a new software. The four basic process; specification, development, validation and evolution are organized differently in different development processes.
## Software specification
Defines the required services and its constraints on the system's operation and development.
Engineering process requirements:
- Elicitation and analysis, what do stakeholders require from the system.
- Specification, defining the requirements in detail.
- Validation, checking the validity of the requirements.
## Design activities
Architectural design, how its components are related, how they are distributed, how are they controlled. (Relations, distribution and control).
Database design, system data structures, (or maybe not even structured).
Interface design, interfaces between system components.(4 types of interfaces??).
Component selection and design, selection of reusable components and if not how will them operate.
## System implementation
## System validation
## Testing stages
Independent tests, or unitary one. System as a whole, emergent properties. Customer testing.
## Software evolution
Inherently changing, requirements change with the business. Irrelevant as fewer and fewer systems are completely new.
### Coping with change
Change is inevitable in all large software systems. Business change, new techniques open up possibilities and platforms are ever changing. Leads to rework (requirements and cost).
#### Reducing rework costs
Anticipation before a significative change is needed. Developing a prototype helps show some key features to the customer. Change tolerance, which may be achieved with incremental development.
# Prototype development
Based on rapid prototyping language tools. Focus on stuff that is not well understood, focus on functional requirements.
## Throw-away prototypes
They are easily degraded over time, impossible to tune the system non-function requirements.
# Incremental delivery
Each one delivers a part of the required functionality.
## Development and delivery
Incremental development:
- Evaluate each increment before processing.
- Normally is used in agile methods.
- Evaluation is done by user/customer proxy.
Incremental delivery:
- Ready to "use" by end-user.
- More realistic about the software usage.
- Difficult as replacement system.
## Advantages
- Will be delivered when functionality is available.
- Act as a prototype to help elicit requirements.
- Lower risk of failure.
- Tends to receive the most testing.
## Problems
- Set of basic facilities used by different parts of the system.
- Specifications is developed in conjunction with the software.
# Process improvement
Reducing costs and accelerate the develop process. Understand existing processes to increase product quality.
## Approaches improvements
Maturity of the process through improvement and project management, introducing good software engineering practice. Reflects the good technical and management practices.
Agile in the form of iterative development and reduction of overheads.
## Activities
- Measurement, attributes to decide if improves have been effective.
- Analysis, assessed also it's weakness bottlenecks identified.
- Change, collect data about the effectiveness of the changes.
## Process measurement
Quantitative data should be collected, if process standards are not set then this is difficult not knowing what to measure. Should be used to make the improvements, yet not based on it as organizational objectives are far more important.