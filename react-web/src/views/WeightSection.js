import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import { BASEURL } from '../Constants';
import { Doughnut } from 'react-chartjs-2';

import Radio from '@material-ui/core/Radio';
import RadioGroup from '@material-ui/core/RadioGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormControl from '@material-ui/core/FormControl';
import FormHelperText from '@material-ui/core/FormHelperText';
import FormLabel from '@material-ui/core/FormLabel';
import Button from '@material-ui/core/Button';
import Input from '@material-ui/core/Input';
import InputLabel from '@material-ui/core/InputLabel';
import InputAdornment from '@material-ui/core/InputAdornment';

const useStyles = makeStyles((theme) => ({
  sectionRoot: {
    margin: theme.spacing(2, 0, 2),
  },
  sectionTitle: {
    margin: theme.spacing(0, 0, 2),
  },
  chartTitle: {
    margin: theme.spacing(2, 0, 2),
  },
  formControl: {
    margin: theme.spacing(3),
  },
  button: {
    margin: theme.spacing(1, 1, 0, 0),
  },
}));

export default function PerformanceSection(props) {
  const classes = useStyles();

  const handleRadioChange = (event) => {
    props.setModel(event.target.value);
  };

  const calculateButtonOnClick = () => {
    props.getWeights(props.selectedModel, props.selectedStocks);
  };

  const handleMoneyInput = (event) => {
    props.setInvestMoney(parseInt(event.target.value));
  };


  return (
    <div className={classes.sectionRoot}>
      <Typography className={classes.sectionTitle} variant="h5">
        Portfolio Weights
      </Typography>
      <Grid container justify="center" alignItems="center" direction="column">
        <FormControl>
          <InputLabel htmlFor="invest-money-input">Invest Money</InputLabel>
          <Input
            id="invest-money-input"
            value={props.investMoney}
            type="number"
            onChange={handleMoneyInput}
            startAdornment={<InputAdornment position="start">$</InputAdornment>}
          />
        </FormControl>
        <FormControl component="fieldset" className={classes.formControl}>
          <FormLabel component="legend">Model select</FormLabel>
          <RadioGroup aria-label="model" name="model" value={props.selectedModel} onChange={handleRadioChange}>
            <FormControlLabel value="basic" control={<Radio />} label="Markowitz" />
            <FormControlLabel value="blacklitterman" control={<Radio />} label="Black litterman" />
          </RadioGroup>
          <Button type="submit" variant="outlined" color="primary" className={classes.button} onClick={calculateButtonOnClick}>
            Save setting
          </Button>
        </FormControl>
      </Grid>
      <Typography className={classes.chartTitle} variant="h5">
        Current Weights
      </Typography>
      <Doughnut data={props.portfolioWeights} />
    </div>
  );
}