import React from 'react';
import Radio from '@material-ui/core/Radio';
import { makeStyles } from '@material-ui/core/styles';
import RadioGroup from '@material-ui/core/RadioGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormControl from '@material-ui/core/FormControl';
import FormLabel from '@material-ui/core/FormLabel';

const useStyles = makeStyles((theme) => ({
  root: {
    textAlign: 'initial',
  }
}));

export default function InvestmentStrategyRadios() {
  const classes = useStyles();

  const [value, setValue] = React.useState('active');

  const handleChange = (event) => {
    setValue(event.target.value);
  };

  return (
    <div className={classes.root}>
      <FormControl component="fieldset">
        <FormLabel component="legend">Investment Strategy</FormLabel>
        <RadioGroup aria-label="Investment Strategy" name="strategy" value={value} onChange={handleChange}>
          <FormControlLabel value="active" control={<Radio />} label="Active" />
          <FormControlLabel value="passive" control={<Radio />} label="Passive" />
        </RadioGroup>
      </FormControl>
    </div>
  );
}